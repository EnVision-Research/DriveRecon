# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.GS_LRM import Gaussian_LRM
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.nn import functional as F
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams,
                 gaussians : GaussianModel,
                 load_iteration = None,
                 resolution_scales = [1.0],
                 train_flag = False,
                 test_flag = False,
                 full_flag = False,
                 #for waymo
                 bg_gaussians: GaussianModel=None, 
                 ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.gaussians = gaussians
        # for waymo
        self.bg_gaussians = bg_gaussians
        self.load_flag = [train_flag, test_flag, full_flag]



        self.train_cameras = {}
        self.test_cameras = {}
        self.full_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            #scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.object_path, n_views=args.n_views, random_init=args.random_init, train_split=args.train_split)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path,"frame_info.json")):
            # print("Found frame_info.json file, assuming Waymo data set!")
            scene_info = sceneLoadTypeCallbacks["Waymo"](args.source_path, args.white_background, args.eval,
                                    use_bg_gs = bg_gaussians is not None,
                                    load_sky_mask = args.load_sky_mask, #False,
                                    load_panoptic_mask = args.load_panoptic_mask, #True,
                                    load_intrinsic = args.load_intrinsic, #False,
                                    load_c2w = args.load_c2w, #False,
                                    load_sam_mask = args.load_sam_mask, #False,
                                    load_semantic_mask = args.load_semantic_mask,
                                    load_Depth_Any= args.load_Depth_Any,
                                    load_dynamic_mask = args.load_dynamic_mask, #False,
                                    load_feat_map = args.load_feat_map, #False,
                                    start_time = args.start_time, #0,
                                    end_time = args.end_time, # 100,
                                    num_pts = args.num_pts,
                                    save_occ_grid = args.save_occ_grid,
                                    occ_voxel_size = args.occ_voxel_size,
                                    recompute_occ_grid = args.recompute_occ_grid,
                                    stride = args.stride,
                                    original_start_time = args.original_start_time,
                                    load_flag = self.load_flag
                                    )
            dataset_type="waymo"
        else:
            assert False, "Could not recognize scene type!"




        self.cameras_extent = scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales:
            #print("Loading Training Cameras")
            if self.load_flag[0]:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            #print("Loading Test Cameras")
            if self.load_flag[1]:
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            #print("Loading Full Cameras")
            if self.load_flag[2]:
                self.full_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.full_cameras, resolution_scale, args)


        self.gaussians.aabb = scene_info.cam_frustum_aabb
        self.gaussians.aabb_tensor = torch.tensor(scene_info.cam_frustum_aabb, dtype=torch.float32)
        self.gaussians.nerf_normalization = scene_info.nerf_normalization
        if train_flag:
            self.gaussians.img_width = scene_info.train_cameras[0].width
            self.gaussians.img_height = scene_info.train_cameras[0].height
        if full_flag:
            self.gaussians.img_width = scene_info.full_cameras[0].width
            self.gaussians.img_height = scene_info.full_cameras[0].height
        if scene_info.occ_grid is not None:
            self.gaussians.occ_grid = torch.tensor(scene_info.occ_grid, dtype=torch.bool)
        else:
            self.gaussians.occ_grid = scene_info.occ_grid
        self.gaussians.occ_voxel_size = args.occ_voxel_size

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            
            # if save_spilt:
            #     pc_dynamic_path = os.path.join(point_cloud_path,"point_cloud_dynamic.ply")
            #     pc_static_path = os.path.join(point_cloud_path,"point_cloud_static.ply")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    #     # background
    #     # if self.gaussians.bg_gaussians is not None:
    #     #     self.gaussians.bg_gs.save_ply(os.path.join(point_cloud_path, "bg_point_cloud.ply"))

    #     if self.bg_gaussians is not None:
    #         self.bg_gaussians.save_ply(os.path.join(point_cloud_path, "bg_point_cloud.ply"))

    def save_gridgs(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}_grid".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # background
        if self.bg_gaussians is not None:
            self.bg_gaussians.save_ply(os.path.join(point_cloud_path, "bg_point_cloud.ply"))


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getFullCameras(self, scale=1.0):
        return self.full_cameras[scale]