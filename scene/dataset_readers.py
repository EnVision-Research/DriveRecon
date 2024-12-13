
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import copy
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from tqdm import trange
from utils.general_utils import PILtoTorch
from tqdm import tqdm
import cv2
from utils.general_utils import sample_on_aabb_surface, get_OccGrid
from utils.segmentation_utils import get_panoptic_id
import torch
from utils.feature_extractor import extract_and_save_features
from utils.image_utils import get_robust_pca

import cv2

from scipy.interpolate import griddata

# IMAGE_LOAD_H = 192
# IMAGE_LOAD_W = 384

IMAGE_LOAD_H = 128
IMAGE_LOAD_W = 256
# IMAGE_LOAD_H = 64
# IMAGE_LOAD_W = 128
Input_scale = 2

def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_input: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    # for waymo
    sky_mask: np.array = None
    depth_map: np.array = None
    depth_any_map: np.array = None
    xyz_map: np.array = None
    time: float = None
    semantic_mask: np.array = None
    instance_mask: np.array = None
    sam_mask: np.array = None
    dynamic_mask: np.array = None
    feat_map: np.array = None
    # grouping
    objects: np.array = None
    #
    intrinsic: np.array = None
    c2w: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    # for waymo
    full_cameras:list
    bg_point_cloud: BasicPointCloud = None
    bg_ply_path: str = None
    cam_frustum_aabb: np.array = None
    num_panoptic_objects: int = 0
    panoptic_id_to_idx: dict = None
    panoptic_object_ids: list = None
    occ_grid: np.array = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)

    green_color = [0, 255, 0]  # [N,3] array
    rgb = np.array([green_color for _ in range(xyz.shape[0])])

    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)




def constructCameras_waymo(frames_list, white_background, mapper = {},
                           load_intrinsic=True, load_c2w=True):
    cam_infos = []

    for idx, frame in enumerate(frames_list):
        # current frame time
        time = mapper[frame["time"]]
        # ------------------
        # load c2w
        # ------------------
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to OpenCV/COLMAP (Y down, Z forward)
        #c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        # ------------------
        # load image
        # ------------------
        cam_name = image_path = frame['file_path']
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0]) # d-nerf 透明背景
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        load_size = frame["load_size"]
        #image = PILtoTorch(image, load_size) #(800,800))
        # resize to load_size
        image = image.resize(load_size, Image.BILINEAR)
        image_input = image.resize([load_size[0] * Input_scale, load_size[1] * Input_scale], Image.BILINEAR)
        # save pil image
        # image.save(os.path.join("debug", image_name + ".png"))
        # ------------------
        # load depth-map
        # ------------------
        depth_map = frame.get('depth_map', None)
        xyz_map = frame.get('xyz_map', None)

        # # # visualize depth map with rgb
        # mask = depth_map > 0
        # # normalize depth map to [0, 255]
        # depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
        # np_depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=1.0), cv2.COLORMAP_JET)
        # # mask empty depth map: depth_map(h,w) , np_depth_map(h,w,3)
        # np_depth_map[~mask] = [255, 255, 255]
        # depth_map_colored = Image.fromarray(np_depth_map)
        # #image_depth = Image.blend(image, depth_map_colored, 0.5)
        # image_np = np.array(image)
        # image_np[mask] = np_depth_map[mask]
        # image_depth = Image.fromarray(image_np)
        # image_depth.save(os.path.join("exp/debug-0", image_name + "_depth.png"))
        # depth_map_colored.save(os.path.join("exp/debug-0", image_name + "_depth_colored.png"))

        # ------------------
        # load depth-any
        # ------------------
        # depth_any_path = image_path.replace("images", "depths")
        # depth_any = Image.open(depth_any_path)
        # depth_any = depth_any.resize(load_size, Image.NEAREST)
        # depth_any = np.array(depth_any)
        depth_any = None
        # ------------------
        # load sky-mask
        # ------------------
        # sky_mask_path, sky_mask = frame["sky_mask_path"], None
        # if sky_mask_path is not None:
        #     sky_mask = Image.open(sky_mask_path)
        #     sky_mask = sky_mask.resize(load_size, Image.BILINEAR)
        sky_mask = None
        # ------------------
        # load intrinsic
        # ------------------
        # intrinsic to fov: intrinsic 已经被 scale
        intrinsic = frame["intrinsic"]
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        # get fov
        fovx = focal2fov(fx, image.size[0])
        fovy = focal2fov(fy, image.size[1])
        FovY = fovy
        FovX = fovx

        # ------------------
        # load sam mask
        # ------------------
        sam_mask_path = image_path.replace("images", "sam_masks").replace("jpg", "png")
        if os.path.exists(sam_mask_path):
            sam_mask = Image.open(sam_mask_path)
            sam_mask = sam_mask.resize(load_size, Image.NEAREST)
            sam_mask = np.array(sam_mask)
            if sam_mask.mean() > 254:
                sam_mask[...] = 0

        # ------------------
        # load semantic mask
        # ------------------
        # DeepLab_V3
        # semantic_mask_path = image_path.replace("images", "DeepLab_V3").replace("jpg", "jpg")
        # semantic_mask = Image.open(semantic_mask_path)
        # semantic_mask = semantic_mask.resize(load_size, Image.NEAREST)
        # semantic_mask = np.array(semantic_mask)
        # if os.path.exists(sam_mask_path):
        #     # semantic_mask[semantic_mask == 1] = 0
        #     half_H = int(IMAGE_LOAD_H/2)
        #     semantic_mask[sam_mask == 255] = 1
        #     semantic_mask[:half_H, :] = np.where((semantic_mask[:half_H, :] == 1) & (sam_mask[:half_H, :] < 255), 0, semantic_mask[:half_H, :])
        # else:
        #     sam_mask = semantic_mask
        # semantic_mask[semantic_mask > 2] = 0

        # Our mask that does not apply to sky seems to be OK,
        # so we do not apply the Sky annotation provided by DeepLab.
        # This is for you to better use our code
        sam_mask[sam_mask > 254] = 1
        semantic_mask = sam_mask


        # ------------------
        # load instance mask
        # ------------------
        instance_mask_path, instance_mask = frame["instance_mask_path"], None
        if instance_mask_path is not None:
            instance_mask = np.load(instance_mask_path)
            instance_mask = Image.fromarray(instance_mask.squeeze(-1))
            instance_mask = instance_mask.resize(load_size, Image.NEAREST)




        # ------------------
        # load dynamic mask
        # ------------------
        dynamic_mask_path, dynamic_mask = frame["dynamic_mask_path"], None
        if dynamic_mask_path is not None:
            dynamic_mask = Image.open(dynamic_mask_path)
            dynamic_mask = dynamic_mask.resize(load_size, Image.NEAREST)

        # ------------------
        # load feat map
        # ------------------
        feat_map_path, feat_map = frame["feat_map_path"], None
        if feat_map_path is not None:
            # mmap_mode="r" is to avoid memory overflow when loading features
            # but it only slightly helps... do we have a better way to load features?
            features = np.load(feat_map_path, mmap_mode="r").squeeze()
            features = torch.from_numpy(features).unsqueeze(0).float()

            # shape: (num_imgs, num_patches_h, num_patches_w, C)
            # featmap_downscale_factor is used to convert the image coordinates to ViT feature coordinates.
            # resizing ViT features to (H, W) using bilinear interpolation is infeasible.
            # imagine a feature array of shape (num_timesteps x num_cams, 640, 960, 768). it's too large to fit in GPU memory.
            featmap_downscale_factor = (
                features.shape[1] / IMAGE_LOAD_H,
                features.shape[2] / IMAGE_LOAD_W,
            )
            # # print(
            #     f"Loaded {features.shape} dinov2_vitb14 features."
            # )
            # # print(f"Feature scale: {featmap_downscale_factor}")
            # # print(f"Computing features PCA...")
            # compute feature visualization matrix
            C = features.shape[-1]
            # no need to compute PCA on the entire set of features, we randomly sample 100k features
            temp_feats = features.reshape(-1, C)
            max_elements_to_compute_pca = min(100000, temp_feats.shape[0])
            selected_features = temp_feats[
                np.random.choice(
                    temp_feats.shape[0], max_elements_to_compute_pca, replace=False
                )
            ]
            target_feature_dim = 3
            device = 'cuda'
            if target_feature_dim is not None:
                # # print(
                #     f"Reducing features to {target_feature_dim} dimensions."
                # )
                # compute PCA to reduce the feature dimension to target_feature_dim
                U, S, reduce_to_target_dim_mat = torch.pca_lowrank(
                    selected_features, q=target_feature_dim, niter=20
                )
                # compute the fraction of variance explained by target_feature_dim
                variances = S**2
                fraction_var_explained = variances / variances.sum()
                # # print(f"[PCA] fraction_var_explained: \n{fraction_var_explained}")
                # # print(
                #     f"[PCA] fraction_var_explained sum: {fraction_var_explained.sum()}",
                # )
                reduce_to_target_dim_mat = reduce_to_target_dim_mat

                # reduce the features to target_feature_dim
                selected_features = selected_features @ reduce_to_target_dim_mat
                features = features @ reduce_to_target_dim_mat
                C = features.shape[-1]

                # normalize the reduced features to [0, 1] along each dimension
                feat_min = features.reshape(-1, C).min(dim=0)[0]
                feat_max = features.reshape(-1, C).max(dim=0)[0]
                features = (features - feat_min) / (feat_max - feat_min)
                selected_features = (selected_features - feat_min) / (feat_max - feat_min)
                feat_min = feat_min.to(device)
                feat_max = feat_max.to(device)
                reduce_to_target_dim_mat = reduce_to_target_dim_mat.to(device)
            # we compute the first 3 principal components of the ViT features as the color
            reduction_mat, feat_color_min, feat_color_max = get_robust_pca(
                selected_features
            )
            # final features are of shape (num_imgs, num_patches_h, num_patches_w, target_feature_dim)
            features = features

            # save visualization parameters
            feat_dimension_reduction_mat = reduction_mat
            feat_color_min = feat_color_min
            feat_color_max = feat_color_max
            del temp_feats, selected_features

            # # print(
            #     f"Feature PCA computed, shape: {feat_dimension_reduction_mat.shape}"
            # )
            # tensor: [91, 137, 64]
            x, y = torch.meshgrid(
                torch.arange(image.size[0]),
                torch.arange(image.size[1]),
                indexing="xy",
            )
            x, y = x.flatten(), y.flatten()
            x, y = x.to(device), y.to(device)

            # we compute the nearest DINO feature for each pixel
            # map (x, y) in the (W, H) space to (x * dino_scale[0], y * dino_scale[1]) in the (W//patch_size, H//patch_size) space
            dino_y = (y * featmap_downscale_factor[0]).long()
            dino_x = (x * featmap_downscale_factor[1]).long()
            # dino_feats are in CPU memory (because they are huge), so we need to move them to GPU
            features = features.squeeze()
            dino_feat = features[dino_y.cpu(), dino_x.cpu()]

            features = dino_feat.reshape(image.size[1], image.size[0], -1)
            feat_map = features.float()
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_input=image_input,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                        # for waymo
                        sky_mask=sky_mask,
                        depth_map=depth_map,
                        depth_any_map=depth_any,
                        xyz_map=xyz_map,
                        time=time,
                        semantic_mask=semantic_mask, instance_mask=instance_mask,
                        sam_mask=sam_mask,
                        dynamic_mask=dynamic_mask,
                        feat_map=feat_map, # [640,960,3]
                        intrinsic=intrinsic if load_intrinsic else None,
                        c2w=c2w if load_c2w else None,
                         ))

    return cam_infos

def readWaymoInfo(path, white_background, eval, extension=".png", use_bg_gs=False,
                  load_sky_mask = False, load_panoptic_mask = True, load_semantic_mask = True,
                  load_sam_mask = False,load_dynamic_mask = False,
                  load_Depth_Any=True,
                  load_feat_map = False,
                  load_intrinsic = True, load_c2w = True,
                  start_time = 0, end_time = -1, num_pts = 5000,
                  save_occ_grid = False, occ_voxel_size = 0.4, recompute_occ_grid=True,
                  stride = 10 , original_start_time = 0, load_flag = [False, False, False],
                  time_interval = 1
                  ):
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    load_size = [IMAGE_LOAD_H, IMAGE_LOAD_W]
    # modified from emer-nerf
    data_root = path
    image_folder = os.path.join(data_root, "images")
    num_seqs = len(os.listdir(image_folder))/5
    start_time = start_time
    if end_time == -1:
        end_time = int(num_seqs)
    else:
        end_time += 1
    camera_list = [1, 0, 2]
    truncated_min_range, truncated_max_range = -10, 70
    cam_frustum_range = [0.01, 80]
    # set img_list
    load_sky_mask = load_sky_mask
    load_semantic_mask = load_semantic_mask
    load_panoptic_mask = load_panoptic_mask
    load_sam_mask = load_sam_mask
    load_dynamic_mask = load_dynamic_mask
    load_feat_map = load_feat_map
    img_filepaths = []
    dynamic_mask_filepaths, sky_mask_filepaths = [], []
    semantic_mask_filepaths, instance_mask_filepaths = [], []
    sam_mask_filepaths = []
    feat_map_filepaths = []
    dynamic_mask_filepaths = []
    lidar_filepaths = []
    for t in range(start_time, end_time, time_interval):
        for cam_idx in camera_list:
            img_filepaths.append(os.path.join(data_root, "images", f"{t:03d}_{cam_idx}.jpg"))
            #dynamic_mask_filepaths.append(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
            sky_mask_filepaths.append(os.path.join(data_root, "sky_masks", f"{t:03d}_{cam_idx}.png"))
            #semantic_mask_filepaths.append(os.path.join(data_root, "semantic_masks", f"{t:03d}_{cam_idx}.png"))
            #instance_mask_filepaths.append(os.path.join(data_root, "instance_masks", f"{t:03d}_{cam_idx}.png"))
            if os.path.exists(os.path.join(data_root, "semantic_segs", f"{t:03d}_{cam_idx}.npy")):
                semantic_mask_filepaths.append(os.path.join(data_root, "semantic_segs", f"{t:03d}_{cam_idx}.npy"))
            else:
                semantic_mask_filepaths.append(None)
            if os.path.exists(os.path.join(data_root, "instance_segs", f"{t:03d}_{cam_idx}.npy")):
                instance_mask_filepaths.append(os.path.join(data_root, "instance_segs", f"{t:03d}_{cam_idx}.npy"))
            else:
                instance_mask_filepaths.append(None)
            if os.path.exists(os.path.join(data_root, "sam_masks", f"{t:03d}_{cam_idx}.jpg")):
                sam_mask_filepaths.append(os.path.join(data_root, "sam_masks", f"{t:03d}_{cam_idx}.jpg"))
            if os.path.exists(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png")):
                dynamic_mask_filepaths.append(os.path.join(data_root, "dynamic_masks", f"{t:03d}_{cam_idx}.png"))
            if load_feat_map:
                feat_map_filepaths.append(os.path.join(data_root, "dinov2_vitb14", f"{t:03d}_{cam_idx}.npy"))
        lidar_filepaths.append(os.path.join(data_root, "lidar", f"{t:03d}.bin"))

    if load_feat_map:
        return_dict = extract_and_save_features(
                input_img_path_list=img_filepaths,
                saved_feat_path_list=feat_map_filepaths,
                img_shape=[644, 966],
                stride=7,
                model_type='dinov2_vitb14',
            )
    img_filepaths = np.array(img_filepaths)
    dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
    sky_mask_filepaths = np.array(sky_mask_filepaths)
    lidar_filepaths = np.array(lidar_filepaths)
    semantic_mask_filepaths = np.array(semantic_mask_filepaths)
    instance_mask_filepaths = np.array(instance_mask_filepaths)
    sam_mask_filepaths = np.array(sam_mask_filepaths)
    feat_map_filepaths = np.array(feat_map_filepaths)
    dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
    # ------------------
    # construct timestamps
    # ------------------
    # original_start_time = 0
    idx_list = range(original_start_time, end_time, time_interval)
    # map time to [0,1]
    timestamp_mapper = {}
    time_line = [i for i in idx_list]
    time_length = end_time - original_start_time - 1
    for index, time in enumerate(time_line):
        timestamp_mapper[time] = (time - original_start_time)/time_length
    # ------------------
    # load poses: intrinsic, c2w, l2w
    # ------------------
    _intrinsics = []
    cam_to_egos = []
    #
    swap_matrix = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]])
    for i in range(len(camera_list)):
        # load intrinsics
        intrinsic = np.loadtxt(os.path.join(data_root, "intrinsics", f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * load_size[1] / ORIGINAL_SIZE[i][1],
            fy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        cx, cy = (
            cx * load_size[1] / ORIGINAL_SIZE[i][1],
            cy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _intrinsics.append(intrinsic)
        # load extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_root, "extrinsics", f"{i}.txt"))
        # ego2cam = ego2cam[[1, 2, 0, 3], :]
        # cam_to_ego = np.linalg.inv(ego2cam)
        # cam_to_ego = cam_to_ego[[1,2,0,3], :]
        # cam_to_ego = swap_matrix @ cam_to_ego
        # opencv coordinate system: x right, y down, z front
        # waymo coordinate system: x front, y left, z up
        # cam_to_egos.append(cam_to_ego)
        cam_to_egos.append(cam_to_ego @ OPENCV2DATASET) # opencv_cam -> waymo_cam -> waymo_ego
    # compute per-image poses and intrinsics
    cam_to_worlds, ego_to_worlds = [], []
    intrinsics, cam_ids = [], []
    lidar_to_worlds = []
    # ===! for waymo, we simplify timestamps as the time indices
    timestamps, timesteps = [], []
    # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
    # the first ego pose as the origin of the world coordinate system.
    ego_to_world_start = np.loadtxt(os.path.join(data_root, "ego_pose", f"{start_time:03d}.txt"))
    # 创建一个交换矩阵

    for t in range(start_time, end_time, time_interval):
        ego_to_world_current = np.loadtxt(os.path.join(data_root, "ego_pose", f"{t:03d}.txt"))
        if end_time-start_time > 10:
            tttt = int(t/3) * 3
            ego_to_world_start = np.loadtxt(os.path.join(data_root, "ego_pose", f"{tttt:03d}.txt"))
        # ego to world transformation: cur_ego -> world -> start_ego(world)
        ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
        ego_to_worlds.append(ego_to_world)
        for cam_id in camera_list:
            cam_ids.append(cam_id)
            # transformation:
            # opencv_cam -> waymo_cam -> waymo_cur_ego -> world -> start_ego(world)
            cam2world = ego_to_world @ cam_to_egos[cam_id] #
            # cam2world = np.dot(cam2world, swap_matrix.T)
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics[cam_id])
            # ===! we use time indices as the timestamp for waymo dataset for simplicity
            # ===! we can use the actual timestamps if needed
            # to be improved
            timestamps.append(t - start_time)
            timesteps.append(t - start_time)
        # lidar to world : lidar = ego in waymo
        lidar_to_worlds.append(ego_to_world)
    # convert to numpy arrays
    intrinsics = np.stack(intrinsics, axis=0)

    cam_to_egos = np.stack(cam_to_egos, axis=0)
    cam_to_worlds = np.stack(cam_to_worlds, axis=0)
    ego_to_worlds = np.stack(ego_to_worlds, axis=0)
    lidar_to_worlds = np.stack(lidar_to_worlds, axis=0)
    cam_ids = np.array(cam_ids)
    timestamps = np.array(timestamps)
    timesteps = np.array(timesteps)
    # ------------------
    # get aabb: c2w --> frunstums --> aabb
    # ------------------
    # compute frustums
    frustums = []
    pix_corners = np.array( # load_size : [h, w]
        [[0, 0], [0, load_size[0]], [load_size[1], load_size[0]], [load_size[1], 0]]
    )
    for c2w, intri in zip(cam_to_worlds, intrinsics):
        frustum = []
        for cam_extent in cam_frustum_range:
            # pix_corners to cam_corners
            cam_corners = np.linalg.inv(intri) @ np.concatenate(
                [pix_corners, np.ones((4, 1))], axis=-1
            ).T * cam_extent
            # cam_corners to world_corners
            world_corners = c2w[:3, :3] @ cam_corners + c2w[:3, 3:4]
            # compute frustum
            frustum.append(world_corners)
        frustum = np.stack(frustum, axis=0)
        frustums.append(frustum)
    frustums = np.stack(frustums, axis=0)
    # compute aabb
    aabbs = []
    for frustum in frustums:
        flatten_frustum = frustum.transpose(0,2,1).reshape(-1,3)
        aabb_min = np.min(flatten_frustum, axis=0)
        aabb_max = np.max(flatten_frustum, axis=0)
        aabb = np.stack([aabb_min, aabb_max], axis=0)
        aabbs.append(aabb)
    aabbs = np.stack(aabbs, axis=0).reshape(-1,3)
    aabb = np.stack([np.min(aabbs, axis=0), np.max(aabbs, axis=0)], axis=0)
    # print('cam frustum aabb min: ', aabb[0])
    # print('cam frustum aabb max: ', aabb[1])
    # ------------------
    # get split: train and test splits from timestamps
    # ------------------
    # mask
    if stride != 0 :
        train_mask = (timestamps % int(stride) != 0) | (timestamps == 0)
    else:
        train_mask = np.ones(len(timestamps), dtype=bool)
    test_mask = ~train_mask
    # mask to index
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    full_idx = np.arange(len(timestamps))
    train_timestamps = timestamps[train_mask]
    test_timestamps = timestamps[test_mask]
    # ------------------
    # load points and depth map
    # ------------------
    pts_path = os.path.join(data_root, "lidar")
    load_lidar, load_depthmap = True, True
    depth_maps = None
    # bg-gs settings
    # use_bg_gs = False
    bg_scale = 2.0 # used to scale fg-aabb
    if not os.path.exists(pts_path) or not load_lidar:
        # random sample
        # Since this data set has no colmap data, we start with random points
        # num_pts = 2000
        # print(f"Generating random point cloud ({num_pts})...")
        aabb_center = (aabb[0] + aabb[1]) / 2
        aabb_size = aabb[1] - aabb[0]
        # We create random points inside the bounds of the synthetic Blender scenes
        random_xyz = np.random.random((num_pts, 3))
        # print('normed xyz min: ', np.min(random_xyz, axis=0))
        # print('normed xyz max: ', np.max(random_xyz, axis=0))
        xyz = random_xyz * aabb_size + aabb[0]
        # print('xyz min: ', np.min(xyz, axis=0))
        # print('xyz max: ', np.max(xyz, axis=0))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        # load lidar points
        origins, directions, points, ranges, laser_ids = [], [], [], [], []
        depth_maps = []
        xyz_maps = []
        w2cs = []
        c2ws = []
        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in range(0, len(lidar_filepaths)):
            # print(lidar_filepaths[t])
            lidar_info = np.memmap(
                lidar_filepaths[t],
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 10)
            #).reshape(-1, 14)
            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length
            lidar_origins = lidar_info[:, :3]
            lidar_points = lidar_info[:, 3:6]
            lidar_ids = lidar_info[:, -1]
            # select lidar points based on a truncated ego-forward-directional range
            # make sure most of lidar points are within the range of the camera
            # valid_mask0 = lidar_points[:, 0] < truncated_max_range
            # valid_mask0 = valid_mask0 & (lidar_points[:, 0] > truncated_min_range)
            # lidar_origins = lidar_origins[valid_mask0]
            # lidar_points = lidar_points[valid_mask0]

            # lidar_ids = lidar_ids[valid_mask0]
            # transform lidar points to world coordinate system
            lidar_origins = (
                lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + lidar_to_worlds[t][:3, 3:4]
            ).T  # lidar_to_worlds几乎没作用

            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D
            # x, y, z = lidar_origins[0,0], lidar_origins[:,1], lidar_origins[:,2]
            # # 创建一个新的3D图形
            # # 创建一个新的3D图形
            # fig = plt.figure(figsize=(12, 8))
            # # 定义视角列表 (方位角, 仰角)
            # view_angles = [
            #     (0, 90),  # 俯视图
            #     (90, 0),  # 右侧视图
            #     (0, 0),  # 前视图
            #     (45, 45),  # 斜视图1
            #     (-60, 30)  # 斜视图2
            # ]
            #
            # # 绘制每个视角的点云
            # for i, (azim, elev) in enumerate(view_angles):
            #     ax = fig.add_subplot(2, 3, i + 1, projection='3d')
            #     ax.scatter(x, y, z, c='r', marker='o')
            #     ax.set_xlabel('X 轴')
            #     ax.set_ylabel('Y 轴')
            #     ax.set_zlabel('Z 轴')
            #     ax.set_title(f'视角 {i + 1}: 方位角={azim}°, 仰角={elev}°')
            #     ax.view_init(elev=elev, azim=azim)
            #
            # # 调整布局并保存图形
            # plt.tight_layout()
            # plt.savefig('point_cloud_multiple_views.png')
            # breakpoint()
            if load_depthmap:
                # transform world-lidar to pixel-depth-map
                for cam_idx in range(len(camera_list)):
                    # world-lidar-pts --> camera-pts : w2c
                    c2w = cam_to_worlds[int(len(camera_list)) * t + cam_idx]
                    w2c = np.linalg.inv(c2w)

                    cam_points = (
                        w2c[:3, :3] @ lidar_origins.T
                        + w2c[:3, 3:4]
                    ).T
                    # camera-pts --> pixel-pts : intrinsic @ (x,y,z) = (u,v,1)*z
                    pixel_points = (
                        intrinsics[int(len(camera_list)) * t + cam_idx] @ cam_points.T
                    ).T
                    # select points in front of the camera
                    Select_Mask = pixel_points[:, 2] > 0
                    pixel_points = pixel_points[pixel_points[:, 2] > 0]
                    # normalize pixel points : (u,v,1)
                    image_points = pixel_points[:, :2] / pixel_points[:, 2:]
                    # filter out points outside the image
                    valid_mask = (
                        (image_points[:, 1] >= 0)
                        & (image_points[:, 1] < load_size[0])
                        & (image_points[:, 0] >= 0)
                        & (image_points[:, 0] < load_size[1])
                    )
                    pixel_points = pixel_points[valid_mask]     # pts_cam : (x,y,z)
                    image_points_temp = copy.deepcopy(image_points) # valid_mask会让一部分点消失
                    image_points = image_points[valid_mask]     # pts_img : (u,v)
                    # compute depth map
                    depth_map = np.zeros(load_size)
                    xyz_map = np.zeros([load_size[0], load_size[1], 3])
                    depth_map[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)] = pixel_points[:, 2]

                    lidar_points_temp1 = lidar_points[Select_Mask]
                    lidar_points_temp2 = lidar_points_temp1[valid_mask]
                    # import scipy.io
                    # scipy.io.savemat(str(cam_idx) + 'lidar_points.mat', {'lidar_points': lidar_points})
                    # scipy.io.savemat(str(cam_idx) + 'lidar_points_temp1.mat', {'lidar_points_temp1': lidar_points_temp1})
                    # scipy.io.savemat(str(cam_idx) + 'lidar_points_temp2.mat', {'lidar_points_temp2': lidar_points_temp2})
                    xyz_map[image_points[:, 1].astype(np.int32),
                            image_points[:, 0].astype(np.int32), 0] = lidar_points_temp2[:, 0]
                    xyz_map[image_points[:, 1].astype(np.int32),
                            image_points[:, 0].astype(np.int32), 1] = lidar_points_temp2[:, 1]
                    xyz_map[image_points[:, 1].astype(np.int32),
                            image_points[:, 0].astype(np.int32), 2] = lidar_points_temp2[:, 2]
                    # for i in range(40):
                    #     image_points = image_points + 16 * np.random.rand(image_points.shape[0], image_points.shape[1]) - 8
                    #     lidar_points_temp2 = lidar_points_temp2 + np.random.rand(lidar_points_temp2.shape[0], lidar_points_temp2.shape[1])
                    #     valid_mask = (
                    #             (image_points[:, 0] >= 0)
                    #             & (image_points[:, 0] < load_size[1])
                    #             & (image_points[:, 1] >= 0)
                    #             & (image_points[:, 1] < load_size[0])
                    #     )
                    #     lidar_points_temp2 = lidar_points_temp2[valid_mask]  # pts_cam : (x,y,z)
                    #     image_points = image_points[valid_mask]  # pts_img : (u,v)
                    #     xyz_map[image_points[:, 1].astype(np.int32),
                    #             image_points[:, 0].astype(np.int32), 0] = lidar_points_temp2[:, 0]
                    #     xyz_map[image_points[:, 1].astype(np.int32),
                    #             image_points[:, 0].astype(np.int32), 1] = lidar_points_temp2[:, 1]
                    #     xyz_map[image_points[:, 1].astype(np.int32),
                    #             image_points[:, 0].astype(np.int32), 2] = lidar_points_temp2[:, 2]
                    # xyz_map0 = copy.deepcopy(xyz_map[:, 0])
                    # xyz_map1 = copy.deepcopy(xyz_map[:, 1])
                    # xyz_map2 = copy.deepcopy(xyz_map[:, 2])
                    # xyz_map0[xyz_map0 <= 0] = 70 * np.random.rand(xyz_map0[xyz_map0<=0].shape[0])
                    # xyz_map1[xyz_map1 <= 0] = 10 * np.random.rand(xyz_map1[xyz_map1 <= 0].shape[0])
                    # xyz_map2[xyz_map2 <= 0] = 2 * np.random.rand(xyz_map2[xyz_map2 <= 0].shape[0])
                    # xyz_map[:, 0] = xyz_map0
                    # xyz_map[:, 1] = xyz_map1
                    # xyz_map[:, 2] = xyz_map2
                    w2cs.append(w2c)
                    c2ws.append(c2w)
                    xyz_maps.append(xyz_map)
                    depth_maps.append(depth_map)



            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = np.linalg.norm(lidar_directions, axis=-1, keepdims=True)
            lidar_directions = lidar_directions / lidar_ranges
            # time indices as timestamp
            #lidar_timestamps = np.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            points.append(lidar_points)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)


        points = np.concatenate(points, axis=0)

        shs = np.random.random((len(points), 3)) / 255.0
        # filter points by cam_aabb
        cam_aabb_mask = np.all((points >= aabb[0]) & (points <= aabb[1]), axis=-1)
        points = points[cam_aabb_mask]
        shs = shs[cam_aabb_mask]

        # downsample points
        points, shs = GridSample3D(points, shs)

        if len(points) > num_pts:
            downsampled_indices = np.random.choice(
                len(points), num_pts, replace=False
            )
            points = points[downsampled_indices]
            shs = shs[downsampled_indices]



        ## 计算 points xyz 的范围
        xyz_min = np.min(points, axis=0)
        xyz_max = np.max(points, axis=0)
        # print("init lidar xyz min:", xyz_min)
        # print("init lidar xyz max:", xyz_max)        # lidar-points aabb (range)

        # save ply
        ply_path = os.path.join(data_root, "ds-points3d.ply")
        # storePly(ply_path, points, SH2RGB(shs) * 255)
        pcd = BasicPointCloud(points=points, colors=SH2RGB(shs), normals=np.zeros((len(points), 3)))

        bg_pcd, bg_ply_path = None, None
        # load depth maps
        if load_depthmap:
            assert depth_maps is not None, "should not use random-init-gs, ans set load_depthmap=True"
            depth_maps = np.stack(depth_maps, axis=0)
            xyz_maps = np.stack(xyz_maps, axis=0)
    # ------------------
    # prepare cam-pose dict
    # ------------------
    train_frames_list = [] # time, transform_matrix(c2w), img_path
    test_frames_list = []
    full_frames_list = []
    if load_flag[0]:
        for idx, t in enumerate(train_timestamps):
            frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引
                                transform_matrix = cam_to_worlds[train_idx[idx]],
                                file_path = img_filepaths[train_idx[idx]],
                                intrinsic = intrinsics[train_idx[idx]],
                                load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                                sky_mask_path = sky_mask_filepaths[train_idx[idx]] if load_sky_mask else None,
                                depth_map = depth_maps[train_idx[idx]] if load_depthmap else None,
                                xyz_map = xyz_maps[train_idx[idx]] if load_depthmap else None,
                                semantic_mask_path = semantic_mask_filepaths[train_idx[idx]] if load_semantic_mask else None,
                                instance_mask_path = instance_mask_filepaths[train_idx[idx]] if load_panoptic_mask else None,
                                sam_mask_path = sam_mask_filepaths[train_idx[idx]] if load_sam_mask else None,
                                feat_map_path = feat_map_filepaths[train_idx[idx]] if load_feat_map else None,
                                dynamic_mask_path = dynamic_mask_filepaths[train_idx[idx]] if load_dynamic_mask else None,
            )
            train_frames_list.append(frame_dict)
    if load_flag[1]:
        for idx, t in enumerate(test_timestamps):
            frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引
                                transform_matrix = cam_to_worlds[test_idx[idx]],
                                file_path = img_filepaths[test_idx[idx]],
                                intrinsic = intrinsics[test_idx[idx]],
                                load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                                sky_mask_path = sky_mask_filepaths[test_idx[idx]] if load_sky_mask else None,
                                depth_map = depth_maps[test_idx[idx]] if load_depthmap else None,
                                xyz_map = xyz_maps[test_idx[idx]] if load_depthmap else None,
                                semantic_mask_path = semantic_mask_filepaths[test_idx[idx]] if load_semantic_mask else None,
                                instance_mask_path = instance_mask_filepaths[test_idx[idx]] if load_panoptic_mask else None,
                                sam_mask_path = sam_mask_filepaths[test_idx[idx]] if load_sam_mask else None,
                                feat_map_path = feat_map_filepaths[test_idx[idx]] if load_feat_map else None,
                                dynamic_mask_path = dynamic_mask_filepaths[test_idx[idx]] if load_dynamic_mask else None,
            )
            test_frames_list.append(frame_dict)
    if load_flag[2]:
        for idx, t in enumerate(timestamps):
            frame_dict = dict(  time = time_line[t+start_time-original_start_time],   # 保存 相对帧索引
                                transform_matrix = cam_to_worlds[full_idx[idx]],
                                file_path = img_filepaths[full_idx[idx]],
                                intrinsic = intrinsics[full_idx[idx]],
                                load_size = [load_size[1], load_size[0]],   # [w, h] for PIL.resize
                                sky_mask_path = sky_mask_filepaths[full_idx[idx]] if load_sky_mask else None,
                                depth_map = depth_maps[full_idx[idx]] if load_depthmap else None,
                                xyz_map = xyz_maps[full_idx[idx]] if load_depthmap else None,
                                semantic_mask_path = semantic_mask_filepaths[full_idx[idx]] if load_semantic_mask else None,
                                instance_mask_path = instance_mask_filepaths[full_idx[idx]] if load_panoptic_mask else None,
                                sam_mask_path = sam_mask_filepaths[full_idx[idx]] if load_sam_mask else None,
                                feat_map_path = feat_map_filepaths[full_idx[idx]] if load_feat_map else None,
                                dynamic_mask_path = dynamic_mask_filepaths[full_idx[idx]] if load_dynamic_mask else None,
            )
            full_frames_list.append(frame_dict)

    # ------------------
    # load cam infos: image, c2w, intrinsic, load_size
    # ------------------
    # print("Reading Training Transforms")
    train_cam_infos = []
    test_cam_infos = []
    full_cam_infos = []
    if load_flag[0]:
        train_cam_infos = constructCameras_waymo(train_frames_list, white_background, timestamp_mapper,
                                                 load_intrinsic=load_intrinsic, load_c2w=load_c2w)
    if load_flag[1]:
        # print("Reading Test Transforms")
        test_cam_infos = constructCameras_waymo(test_frames_list, white_background, timestamp_mapper,
                                                load_intrinsic=load_intrinsic, load_c2w=load_c2w)
    if load_flag[2]:
        # print("Reading Full Transforms")
        full_cam_infos = constructCameras_waymo(full_frames_list, white_background, timestamp_mapper,
                                                load_intrinsic=load_intrinsic, load_c2w=load_c2w)

    ## print("Generating Video Transforms")
    #video_cam_infos = generateCamerasFromTransforms_waymo(test_frames_list, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    if load_flag[0]:
        nerf_normalization = getNerfppNorm(train_cam_infos)
    if load_flag[2]:
        nerf_normalization = getNerfppNorm(full_cam_infos)


    # ------------------
    # find panoptic-objec numbers
    # ------------------
    num_panoptic_objects = 0
    panoptic_object_ids = None
    panoptic_id_to_idx = {}
    if load_panoptic_mask:
        panoptic_object_ids_list = []
        for cam in train_cam_infos + test_cam_infos:
            if cam.semantic_mask is not None and cam.instance_mask is not None:
                panoptic_object_ids = get_panoptic_id(cam.semantic_mask, cam.instance_mask).unique()
                panoptic_object_ids_list.append(panoptic_object_ids)
        # get unique panoptic_objects_ids
        panoptic_object_ids = torch.cat(panoptic_object_ids_list).unique().sort()[0].tolist()
        num_panoptic_objects = len(panoptic_object_ids)
        # map panoptic_id to idx
        for idx, panoptic_id in enumerate(panoptic_object_ids):
            panoptic_id_to_idx[panoptic_id] = idx
    scene_info = SceneInfo(point_cloud=pcd,
                           bg_point_cloud=bg_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           full_cameras=full_cam_infos,
                           #video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           # background settings
                           ply_path=pts_path,
                           bg_ply_path=bg_ply_path,
                           cam_frustum_aabb=aabb,
                           # panoptic segs
                           num_panoptic_objects=num_panoptic_objects,
                           panoptic_object_ids=panoptic_object_ids,
                           panoptic_id_to_idx=panoptic_id_to_idx,
                           # occ grid
                           occ_grid=occ_grid if save_occ_grid else None,
                           )
    return scene_info


sceneLoadTypeCallbacks = {
    "Waymo" : readWaymoInfo,
}

def GridSample3D(in_pc,in_shs, voxel_size=0.013):
    in_pc_ = in_pc[:,:3].copy()
    quantized_pc = np.around(in_pc_ / voxel_size)
    quantized_pc -= np.min(quantized_pc, axis=0)
    pc_boundary = np.max(quantized_pc, axis=0) - np.min(quantized_pc, axis=0)

    voxel_index = quantized_pc[:,0] * pc_boundary[1] * pc_boundary[2] + quantized_pc[:, 1] * pc_boundary[2] + quantized_pc[:,2]

    split_point, index = get_split_point(voxel_index)

    in_points = in_pc[index,:]
    out_points = in_points[split_point[:-1], :]

    in_colors = in_shs[index]
    out_colors = in_colors[split_point[:-1]]

    # 创建一个新的BasicPointCloud实例作为输出
    # out_pc =out_points
    # #remap index in_pc to out_pc
    # remap = np.zeros(in_pc.points.shape[0])

    # for ind in range(len(split_point)-1):
    #     cur_start = split_point[ind]
    #     cur_end = split_point[ind+1]
    #     remap[cur_start:cur_end] = ind

    # remap_back = remap.copy()
    # remap_back[index] = remap

    # remap_back = remap_back.astype(np.int64)
    return out_points, out_colors

def get_split_point(labels):
    index = np.argsort(labels)
    label = labels[index]
    label_shift = label.copy()

    label_shift[1:] = label[:-1]
    remain = label - label_shift
    step_index = np.where(remain > 0)[0].tolist()
    step_index.insert(0,0)
    step_index.append(labels.shape[0])
    return step_index,index
