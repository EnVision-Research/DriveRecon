#
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
import gc
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, compute_depth
from gaussian_renderer import LRM_render, network_gui
import sys
from scene import Scene, GaussianModel, Gaussian_LRM
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
# import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from torch import nn
import scipy.io

import numpy as np
import time
import json
from utils.video_utils import render_pixels, save_videos
from utils.visualization_tools import compute_optical_flow_and_save
from scene.gaussian_model import merge_models
from scipy.interpolate import CubicSpline
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
TENSORBOARD_FOUND = False

current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

render_keys = [
    "gt_rgbs",
    "rgbs",
    "depths",
    "dynamic_rgbs",
    "static_rgbs",
    # "forward_flows",
    # "backward_flows",
]


def depth2xyz(Depth, intrinsics, c2ws, N=3, H=640, W=960):

    channel_1 = torch.arange(W).unsqueeze(0).expand(H, -1).repeat(N, 1, 1).to(Depth.device)
    channel_2 = torch.arange(H).unsqueeze(0).expand(W, -1).permute(1, 0).repeat(N, 1, 1).to(Depth.device)
    uv_map = torch.stack((channel_1, channel_2, Depth), dim=-1)  # 三张图，H， W， u， v， depth
    cam_map = torch.zeros_like(uv_map)
    cam_map[..., 2] += uv_map[..., 2]
    cam_map[..., 0:2] += torch.mul(uv_map[..., 0:2], uv_map[..., 2].unsqueeze(-1))
    cam_map = cam_map.reshape(N, -1, 3)
    xyzs = torch.zeros_like(cam_map)
    xyzs_world = torch.zeros_like(cam_map)
    for index in range(N):
        intrinsic, c2w = intrinsics[index], c2ws[index]
        xyzs[index, ] += torch.mm(torch.inverse(intrinsic.to(cam_map.device)), cam_map[index,].T).T
        temp_one = torch.ones_like(xyzs[index, :, 0]).unsqueeze(-1)
        temp = torch.cat((xyzs[index], temp_one), dim=-1)
        temp = torch.mm(c2w.to(uv_map.device), temp.T)
        xyzs_world[index] += temp[0:3, :].T
    xyzs = xyzs_world.reshape(-1, 3)
    return xyzs

@torch.no_grad()
def do_evaluation(
        viewpoint_stack_full,
        viewpoint_stack_test,
        viewpoint_stack_train,
        gaussians,
        bg,
        pipe,
        LRM_render,
        eval_dir,
        render_full,
        step: int = 0,
        args=None,
):
    if len(viewpoint_stack_test) != 0:
        print("Evaluating Test Set Pixels...")
        render_results = render_pixels(
            viewpoint_stack_test,
            gaussians,
            bg,
            pipe,
            LRM_render,
            compute_metrics=True,
            return_decomposition=True,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/test/{k}"] = v

        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/test_videos", exist_ok=True)

        test_metrics_file = f"{eval_dir}/metrics/{step}_images_test_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        video_output_pth = f"{eval_dir}/test_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_test) // 3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    if len(viewpoint_stack_train) != 0 and len(viewpoint_stack_test) != 0:
        print("Evaluating train Set Pixels...")
        render_results = render_pixels(
            viewpoint_stack_train,
            gaussians,
            bg,
            pipe,
            LRM_render,
            compute_metrics=True,
            return_decomposition=False,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/train/{k}"] = v

        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/train_videos", exist_ok=True)

        train_metrics_file = f"{eval_dir}/metrics/{step}_images_train.json"
        with open(train_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {train_metrics_file}")

        video_output_pth = f"{eval_dir}/train_videos/{step}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_train) // 3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results
        torch.cuda.empty_cache()

    if render_full:
        print("Evaluating Full Set...")
        render_results = render_pixels(
            viewpoint_stack_full,
            gaussians,
            bg,
            pipe,
            LRM_render,
            compute_metrics=True,
            return_decomposition=True,
            debug=args.debug_test
        )
        eval_dict = {}
        for k, v in render_results.items():
            if k in [
                "psnr",
                "ssim",
                "lpips",
                # "feat_psnr",
                "masked_psnr",
                "masked_ssim",
                # "masked_feat_psnr",
            ]:
                eval_dict[f"pixel_metrics/full/{k}"] = v

        os.makedirs(f"{eval_dir}/metrics", exist_ok=True)
        os.makedirs(f"{eval_dir}/full_videos", exist_ok=True)

        test_metrics_file = f"{eval_dir}/metrics/{step}_images_full_{current_time}.json"
        with open(test_metrics_file, "w") as f:
            json.dump(eval_dict, f)
        print(f"Image evaluation metrics saved to {test_metrics_file}")

        # if render_video_postfix is None:
        video_output_pth = f"{eval_dir}/full_videos/{step}.mp4"
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=int(len(viewpoint_stack_full) // 3),
            keys=render_keys,
            num_cams=3,
            save_seperate_video=True,
            fps=24,
            verbose=True,
        )

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0
    Densification_flag = 0
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack:
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
        pop_indexex = [i for i in range(len(viewpoint_stack))]

    batch_size = opt.batch_size
    print("data loading done")


    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        # batch size
        idx = 0
        viewpoint_cams = []
        viewpoint_cams_input = []
        while idx < batch_size:
            pop_int = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cam = viewpoint_stack.pop(pop_int)
            pop_index = pop_indexex.pop(pop_int)
            pop_index1 = int(pop_index/3)
            sub_index = pop_index % 3
            if not viewpoint_stack:
                viewpoint_stack = temp_list.copy()
                pop_indexex = [i for i in range(len(viewpoint_stack))]
            viewpoint_cams.append(viewpoint_cam)
            viewpoint_cams_input.append([temp_list[pop_index1*3],
                                         temp_list[pop_index1*3+1],
                                         temp_list[pop_index1*3+2]])
            idx += 1
        if len(viewpoint_cams) == 0:
            continue
        # print(len(viewpoint_cams))
        # breakpoint()
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        depth_preds = []
        gt_depths = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        # breakpoint()

        for cams_input, viewpoint_cam in zip(viewpoint_cams_input, viewpoint_cams):
            if dataset.lrm:
                cams = []
                xyzs = []
                intrinsics = []
                c2ws = []
                for cam in cams_input:
                    cams.append(cam.original_image.cuda())
                    xyzs.append(cam.xyz_map.cuda())
                    intrinsics.append(cam.intrinsic.cuda())
                    c2ws.append(cam.c2w.cuda())
                cams = torch.stack(cams, dim=0)  # [3, 3, 640, 960]
                N, C, H, W = cams.shape # N为视角的数量
                means3D, scale, rotation, opacity, rgbs_or_shs = gaussians.forward_gaussians(cams, intrinsics, c2ws)
                if dataset.shs_pre:
                    rgbs_or_shs = rgbs_or_shs.reshape(-1, (3 + 1) ** 2, 3)
                else:
                    rgbs_or_shs = rgbs_or_shs.reshape(-1, 3)
                means3D0, scale0, rotation0, opacity0, rgbs_or_shs0 = torch.squeeze(means3D), torch.squeeze(scale), \
                                                                          torch.squeeze(rotation), opacity.squeeze(0), \
                                                                          torch.squeeze(rgbs_or_shs)
                Depth_pre = means3D.reshape(N, H, W, 3)
                Depth_pre = 255.0 * torch.mean(Depth_pre, dim=-1)  # 方便代码的通用性，既可以预测xyz，又可以预测depth的平均值
                xyz_world = depth2xyz(Depth_pre, intrinsics, c2ws, N=N, H=H, W=W)
                time0 = torch.tensor(viewpoint_cam.time).to(means3D.device).repeat(means3D.shape[1], 1)
                screenspace_points = torch.zeros_like(means3D0, dtype=torch.float32, requires_grad=True,
                                                      device=means3D.device)
                try:
                    screenspace_points.retain_grad()
                except:
                    pass
                means2D0 = screenspace_points

                Guassian_para = {
                    'means3D':  xyz_world,  # weight * xyz_world + (1 - weight) * xyzs0, # (iteration/5000) * means3D0 + (1 - iteration/5000) * xyzs0, # (iteration/5000) * means3D0 + (1 - iteration/5000) *xyzs
                    'scale': scale0,  # [0:means3D.shape[0]]
                    'rotation': rotation0,
                    'opacity': opacity0,
                    'rgbs_or_shs': rgbs_or_shs0,
                    'means2D': means2D0,
                    'time': time0,
                }
            else:
                # 这里是原始3D
                dataset.lrm = False
                means3D = gaussians.get_xyz
                opacity = gaussians._opacity
                rgbs_or_shs = gaussians.get_features
                scale = gaussians._scaling
                rotation = gaussians._rotation
                screenspace_points = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=True,
                                                      device=means3D.device)
                # xyz = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=True,
                #                                       device=means3D.device)
                try:
                    screenspace_points.retain_grad()
                except:
                    pass
                means2D = screenspace_points
                time = torch.tensor(viewpoint_cam.time).to(means3D.device).repeat(means3D.shape[0], 1)  # 这里？
                Guassian_para = {
                    'means3D': means3D,
                    'scale': scale,
                    'rotation': rotation,
                    'opacity': opacity,
                    'rgbs_or_shs': rgbs_or_shs,
                    'means2D': means2D,
                    'time': time,
                }


            render_pkg = LRM_render(Guassian_para, viewpoint_cam, gaussians, pipe, background, stage=stage,
                                    return_dx=True,
                                    render_feat=True if ('fine' in stage and args.feat_head) else False,
                                    sh_flag=dataset.shs_pre, LRM_flag=dataset.lrm)


            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth_pred = render_pkg["depth"]
            depth_pred_flag = (depth_preds != None)
            if depth_pred_flag:
                depth_preds.append(depth_pred.unsqueeze(0))
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth = viewpoint_cam.depth_map.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        # breakpoint()
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        if depth_pred_flag:
            depth_pred_tensor = torch.cat(depth_preds, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        gt_depth_tensor = torch.cat(gt_depths, 0).float()
        if iteration % 4000==0:
            from PIL import Image
            gt_RGB = Image.fromarray(np.array(gt_image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0, dtype=np.byte),
                                     "RGB")
            pre_RGB = Image.fromarray(np.array(image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0, dtype=np.byte), "RGB")
            gt_RGB.save(os.path.join("exp/debug-0", str(iteration) + '_' + str(pop_int) + "gt.png"))
            pre_RGB.save(os.path.join("exp/debug-0", str(iteration) + '_' + str(pop_int) + "pre.png"))

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])  # + xyz_loss/xyz_loss.item()
        loss = Ll1
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        if Densification_flag:
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        iter_end.record()

        if iteration < final_iter + 1:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]

            print_dict = {
                "step": f"{iteration}",
                "Loss": f"{ema_loss_for_log:.{7}f}",
                "psnr": f"{psnr_:.{2}f}",
            }
            print(print_dict)

            if iteration % 100 == 0:
                dynamic_points = 0
                if 'fine' in stage and not args.no_dx:
                    dx_abs = torch.abs(render_pkg['dx'])  # [N,3]
                    max_values = torch.max(dx_abs, dim=1)[0]  # [N]
                    thre = torch.mean(max_values)
                    mask = (max_values > thre)
                    dynamic_points = torch.sum(mask).item()

                print_dict = {
                    "step": f"{iteration}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "psnr": f"{psnr_:.{2}f}",
                    "dynamic point": f"{dynamic_points}",
                    "point": f"{total_point}",
                }
                progress_bar.set_postfix(print_dict)
                metrics_file = f"{scene.model_path}/logger.json"
                with open(metrics_file, "a") as f:
                    json.dump(print_dict, f)
                    f.write('\n')

                progress_bar.update(100)
            if iteration == final_iter:
                progress_bar.close()

            # Log and save
            timer.pause()

            timer.start()
            # Densification


            if (iteration in checkpoint_iterations):
                save_path = "chkpnt" + f"_{stage}_" + str(30000) + ".pth"
                for file in os.listdir(scene.model_path):
                    if file.endswith(".pth") and file != save_path:
                        os.remove(os.path.join(scene.model_path, file))

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth")




def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = Gaussian_LRM(dataset.sh_degree, hyper, dataset.shs_pre)
    dataset.model_path = args.model_path
    timer = Timer()
    timer.start()
    # eval
    eval_dir = os.path.join(args.model_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    dataset.start_time = 0
    dataset.end_time = 1
    scene = Scene(dataset, gaussians, load_coarse=None)
    viewpoint_stack_full = scene.getFullCameras().copy()
    viewpoint_stack_test = scene.getTestCameras().copy()
    viewpoint_stack_train = scene.getTrainCameras().copy()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians.training_setup(opt)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer)




def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = None
        # tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if len(scene.getTestCameras()) != 0:
            validation_configs = ({'name': 'test',
                                   'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in
                                               range(10, 5000, 299)]},
                                  {'name': 'train',
                                   'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                               range(10, 5000, 299)]})
        else:
            validation_configs = ({'name': 'train',
                                   'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                               range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, stage=stage, *renderArgs)["render"], 0.0,
                                        1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage + "/" + config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage + "/" + config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                    gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask

                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate',
                                 scene.gaussians._deformation_table.sum() / scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram",
                                    scene.gaussians._deformation_accum.mean(dim=-1) / 100, iteration, max_bins=500)

        torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7000, 14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int,
                        default=[10_000, 20_000, 30_000, 40_000, 50_000, 20_000, 50_0000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="waymo")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--eval_only", action="store_true", help="perform evaluation only")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--prior_checkpoint", type=str, default=None)
    parser.add_argument("--merge", action="store_true", help="merge gaussians")
    parser.add_argument("--prior_checkpoint2", type=str, default=None)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
