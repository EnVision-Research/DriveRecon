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
from typing import Callable, Dict, List, Optional
from torch import Tensor
from lpipsPyTorch import lpips

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
from accelerate import Accelerator, DistributedDataParallelKwargs

import numpy as np
import time
import json
from utils.video_utils import render_pixels, save_videos
from utils.visualization_tools import compute_optical_flow_and_save
from scene.gaussian_model import merge_models
from scipy.interpolate import CubicSpline
from scene.dataset import WaymoDataset
from utils.image_utils import psnr
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from skimage.metrics import structural_similarity as ssim
import open3d as o3d


to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)
get_numpy: Callable[[Tensor], np.ndarray] = lambda x: x.squeeze().detach().cpu().numpy()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# machine_rank = int(os.environ['VC_TASK_INDEX'])

current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def custom_collate_fn(batch):
    # 处理自定义类对象，假设每个 batch 项目都是一个 Scene 类对象
    scenes = [item for item in batch]
    # 你可以在这里添加更多处理逻辑，比如提取特定属性并转换为 tensor
    return scenes

# def visual_result(resulut):
#     rgbs_gt = []
#     rgbs_pre = []
#     depth_pre = []
#     seg_pre = []
#     rgbs_dy = []
#     rgbs_static = []
#     for result in resulut:
#         view_num = len(result['image_gts'])
#         for index_image in range(view_num):
#             rgbs_gt = result['image_gts'][index_image].detach().cpu().numpy() # [3, 128, 256]
#             rgbs_pre = result['image_recstuctions'][index_image].detach().cpu().numpy() # [3, 128, 256]
#             depth_pre = result['depth_recstuctions'][index_image].detach().cpu().numpy() # [1, 128, 256]
#             surf_pre = result['surf_recstuctions'][index_image].detach().cpu().numpy() # [3, 128, 256]
#             normal_pre = result['normal_recstuctions'][index_image].detach().cpu().numpy() # [3, 128, 256]
#             sem = result['semantic_pre'][index_image].argmax(dim=-1)
#             seg_pre = sem.detach().cpu().numpy() # [128, 256]
#             image_dy = result['image_dy'][index_image].detach().cpu().numpy() # [3, 128, 256]
#             image_static = result['image_static'][index_image].detach().cpu().numpy() # [3, 128, 256]
#             breakpoint()

import cv2
import numpy as np

def visual_result(results, output_video_path='./exp/1/', frame_rate=10):
    # 假设每个图像的大小为 [3, 128, 256]
    h, w = 128, 256
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
    # 根据视角数量调整视频帧的宽度
    view_num = 3
    frame_width = w * view_num
    frame_height = h * 6  # 每个属性在垂直方向上堆叠

    index = 0
    all_views = []
    for result in results:
        # 列表来存储每个视角的所有属性
        for index_image in range(view_num):
            # 提取并转置图像
            rgb_gt = np.transpose(result['image_gts'][index_image].detach().cpu().numpy(), (1, 2, 0)) * 255
            rgb_pre = np.transpose(result['image_recstuctions'][index_image].detach().cpu().numpy(), (1, 2, 0)) * 255
            depth_pre = result['depth_recstuctions'][index_image].detach().cpu().numpy() # [128, 256]
            depth_pre = cv2.cvtColor((depth_pre / np.max(depth_pre) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            surf_pre = np.transpose(result['surf_recstuctions'][index_image].detach().cpu().numpy(), (1, 2, 0)) * 255
            normal_pre = np.transpose(result['normal_recstuctions'][index_image].float().detach().cpu().numpy(), (1, 2, 0)) * 255
            # seg_pre = cv2.applyColorMap((result['semantic_pre'][index_image].argmax(dim=-1).detach().cpu().numpy() * 40).astype(np.uint8), cv2.COLORMAP_JET)
            seg_pre = cv2.applyColorMap(
                (result['semantic_mask'][index_image].float().detach().cpu().numpy() * 40).astype(np.uint8),
                cv2.COLORMAP_JET)
            image_dy = np.transpose(result['image_dy'][index_image].detach().cpu().numpy(), (1, 2, 0)) * 255
            image_static = np.transpose(result['image_static'][index_image].detach().cpu().numpy(), (1, 2, 0)) * 255
            # 转换 RGB 到 BGR
            rgb_gt = cv2.cvtColor(rgb_gt.astype(np.uint8), cv2.COLOR_RGB2BGR)
            rgb_pre = cv2.cvtColor(rgb_pre.astype(np.uint8), cv2.COLOR_RGB2BGR)
            image_dy = cv2.cvtColor(image_dy.astype(np.uint8), cv2.COLOR_RGB2BGR)
            image_static = cv2.cvtColor(image_static.astype(np.uint8), cv2.COLOR_RGB2BGR)
            # 将单个视角的所有图像在垂直方向上拼接
            single_view_combined = np.vstack((rgb_gt, rgb_pre, depth_pre, seg_pre, image_dy, image_static)).astype(np.uint8)
            if index > 0 and index%view_num == 0:
                # 将所有视角在水平方向上拼接
                frame_combined = np.hstack(all_views)
                cv2.imwrite(output_video_path + f'Frame_{1000 + index}.png', frame_combined.astype(np.uint8))
                all_views = []
            index = index + 1
            all_views.append(single_view_combined)




def eval_result(resulut):
    psnrs = []
    ssim_scores = []
    lpipss = []

    psnrs_d = []
    ssim_scores_d = []

    psnrs_s = []
    ssim_scores_s = []
    for result in resulut:
        for index_image in range(len(result['image_gts'])):
            image_gts = result['image_gts'][index_image]
            image_recstuctions = result['image_recstuctions'][index_image]
            depth_recstuctions = result['depth_recstuctions'][index_image]
            surf_recstuctions = result['surf_recstuctions'][index_image]
            normal_recstuctions = result['normal_recstuctions'][index_image]
            semantic_mask = result['semantic_mask'][index_image]
            semantic_pre = result['semantic_pre'][index_image]

            static_mask = semantic_mask != 1
            dy_mask = semantic_mask == 1
            # static_mask = static_mask.repeat(3, 1, 1)
            # dy_mask = dy_mask.repeat(3, 1, 1)

            image_gts_s = image_gts.permute(1, 2, 0)[static_mask]
            image_gts_d = image_gts.permute(1, 2, 0)[dy_mask]
            image_recstuctions_s = image_recstuctions.permute(1, 2, 0)[static_mask]
            image_recstuctions_d = image_recstuctions.permute(1, 2, 0)[dy_mask]
            image_gts_s = image_gts_s.permute(1, 0)
            image_gts_d = image_gts_d.permute(1, 0)
            image_recstuctions_s = image_recstuctions_s.permute(1, 0)
            image_recstuctions_d = image_recstuctions_d.permute(1, 0)

            psnrs.append(psnr(image_recstuctions, image_gts).mean().double().item())
            # ssim_scores.append(ssim(rgb, gt_rgb).mean().item())
            ssim_scores.append(
                ssim(
                    get_numpy(image_recstuctions),
                    get_numpy(image_gts),
                    data_range=1.0,
                    channel_axis=0,
                )
            )
            lpipss.append(torch.tensor(lpips(image_recstuctions, image_gts, net_type='alex')).mean().item())

            psnrs_s.append(psnr(image_recstuctions_s, image_gts_s).mean().double().item())
            # ssim_scores.append(ssim(rgb, gt_rgb).mean().item())
            ssim_scores_s.append(
                ssim(
                    get_numpy(image_recstuctions.permute(1, 2, 0)),
                    get_numpy(image_gts.permute(1, 2, 0)),
                    data_range=1.0,
                    channel_axis=-1,
                    full=True,
                )[1][static_mask].mean()
            )

            psnrs_d.append(psnr(image_recstuctions_d, image_gts_d).mean().double().item())
            # ssim_scores.append(ssim(rgb, gt_rgb).mean().item())
            ssim_scores_d.append(
                ssim(
                    get_numpy(image_recstuctions.permute(1, 2, 0)),
                    get_numpy(image_gts.permute(1, 2, 0)),
                    data_range=1.0,
                    channel_axis=-1,
                    full=True,
                )[1][dy_mask].mean()
            )


    psnrs, ssim_scores, lpipss, psnrs_s, ssim_scores_s, psnrs_d, ssim_scores_d =\
    np.nanmean(np.array(psnrs)), np.nanmean(np.array(ssim_scores)), np.nanmean(np.array(lpipss)), \
    np.nanmean(np.array(psnrs_s)), np.nanmean(np.array(ssim_scores_s)), np.nanmean(np.array(psnrs_d)), \
    np.nanmean(np.array(psnrs_d))
    return [psnrs, ssim_scores, lpipss, psnrs_s, ssim_scores_s, psnrs_d, ssim_scores_d]

def evaling(dataset, hyper, opt, pipe, checkpoint):
    ## eval
    time_length_load = 196# 6
    ## Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1
    )

    stage = "coarse"
    # model
    gaussians = Gaussian_LRM(dataset.sh_degree, hyper, View_Num=3, shs_pre=dataset.shs_pre)
    dataset.model_path = args.model_path
    train_iter = opt.coarse_iterations
    if checkpoint:
            model_params = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    eval_dir = os.path.join(args.model_path, "eval")
    time_length = args.time_lenth
    View_Num = args.veiw_num
    rendered_num = args.rendered_num
    os.makedirs(eval_dir, exist_ok=True)
    file_root = 'data/waymo/NOTA2_64/training/'
    _WaymoDataset = WaymoDataset(file_root, dataset, gaussians,
                                 time_lenth=time_length_load,
                                 load_coarse=None,
                                 Train_flag=False,
                                 Full_flag=True)
    batch_size = 1
    train_dataloader = torch.utils.data.DataLoader(
        _WaymoDataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    timer = Timer()
    timer.start()


    gaussians.training_setup(opt)

    # option
    gaussians, gaussians.optimizer, train_dataloader = accelerator.prepare(gaussians, gaussians.optimizer, train_dataloader)


    # background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    first_iter = 0
    # progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    iteration_real = 0
    matric = []
    matric_noval = []
    for j, scenes in enumerate(train_dataloader):
        with accelerator.accumulate(gaussians):
            # 读取多个场景流的多个视角
            viewpoint_stack = []
            for scene in scenes: # batchsize个场景
                train_cams = scene.getFullCameras()
                for train_cam in train_cams: # 多个时间多个视角
                    viewpoint_stack.append(train_cam) # time_length*左中右

            # 随机选取视角
            cams = []
            cams_input = []
            xyzs = []
            depths = []
            depth_anys = []
            intrinsics = []
            c2ws = []
            semantic_mask = []
            #### 整理输入和辅助监督
            for view_index in range(len(viewpoint_stack)):
                cam = viewpoint_stack[view_index]
                cams.append(cam.original_image)
                cams_input.append(cam.image_input)
                xyzs.append(cam.xyz_map)
                depths.append(cam.depth_map)
                depth_anys.append(cam.depth_any_map)
                semantic_mask.append(cam.semantic_mask)
                intrinsics.append(cam.intrinsic)
                c2ws.append(cam.c2w)
            # 把batch size, Tmeper放到前面来
            cams = torch.stack(cams, dim=0).reshape(batch_size, time_length_load, View_Num, 3, cams[0].shape[-2],
                                                    cams[0].shape[-1]).to(torch.bfloat16)
            semantic_mask = torch.stack(semantic_mask, dim=0).reshape(batch_size, time_length_load, View_Num, cams[0].shape[-2],
                                                    cams[0].shape[-1]).to(torch.bfloat16)
            cams_input = torch.stack(cams_input, dim=0).reshape(batch_size, time_length_load, View_Num, 3,
                                                                cams_input[0].shape[-2],
                                                                cams_input[0].shape[-1]).to(torch.bfloat16)
            xyzs = torch.stack(xyzs, dim=0).reshape(batch_size, time_length_load, View_Num, 3, xyzs[0].shape[-2],
                                                    xyzs[0].shape[-1]).to(torch.bfloat16)
            depths = torch.stack(depths, dim=0).reshape(batch_size, time_length_load, View_Num, 1, xyzs[0].shape[-2],
                                                        xyzs[0].shape[-1]).to(torch.bfloat16)
            depth_anys = torch.stack(depth_anys, dim=0).reshape(batch_size, time_length_load, View_Num, 1, xyzs[0].shape[-2],
                                                                xyzs[0].shape[-1]).to(torch.bfloat16)
            intrinsics = torch.stack(intrinsics, dim=0).reshape(batch_size, time_length_load, View_Num, 3, 3).to(torch.bfloat16)  #
            c2ws = torch.stack(c2ws, dim=0).reshape(batch_size, time_length_load, View_Num, 4, 4).to(torch.bfloat16)
            # cams[0, 2, 2]-viewpoint_stack[4*3*0+3*2+2].original_image






            noval_stride = 10
            _step = int((time_length_load - time_length + 1 - 2) / noval_stride)
            result_diction_noval = []
            for i in range(1, _step + 1):
                input_index = [i * noval_stride - 2, i * noval_stride -1, i * noval_stride + 1]
                rendered_index = [i * noval_stride * View_Num, i * noval_stride * View_Num + View_Num]
                result_diction_sub = gaussians.noval_view(cams_input[:, input_index, ...],
                                                     cams[:, input_index, ...],
                                                     xyzs[:, input_index, ...],
                                                     depths[:, input_index, ...],
                                                     depth_anys[:, input_index, ...],
                                                     semantic_mask[:, input_index, ...],
                                                     intrinsics[:, input_index, ...],
                                                     c2ws[:, input_index, ...],
                                                     viewpoint_stack[i * noval_stride * View_Num: i * noval_stride * View_Num + View_Num],
                                                     pipe=pipe,
                                                     background=background, stage=stage,
                                                     return_dx=False, render_feat=False,
                                                     sh_flag=dataset.shs_pre)
                result_diction_noval.append(result_diction_sub)

            result_m_noval = eval_result(result_diction_noval)
            print("*********noval******")
            print(result_m_noval)
            matric_noval.append(result_m_noval)
            matric_np_noval = np.array(matric_noval)
            print(np.nanmean(matric_np_noval, axis=1))
            print(np.mean(matric_np_noval, axis=0))

            iteration_real += 1
            if iteration_real>64:
                break














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
                        default=[100, 1_000, 2_000, 5_000, 10_000, 20_000,
                                 50_000, 100_000, 150_000, 300_000, 500_000,
                                 700_000, 800_000, 900_000,
                                 1_000_000])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--expname", type=str, default="waymo")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--eval_only", action="store_true", help="perform evaluation only")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--prior_checkpoint", type=str, default=None)
    parser.add_argument("--merge", action="store_true", help="merge gaussians")
    parser.add_argument("--prior_checkpoint2", type=str, default=None)
    parser.add_argument("--eval_time_length", type=int, default=200)


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


    evaling(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.checkpoint_path)

    # All done
    print("\nTraining complete.")
# python eval.py --checkpoint_path "./checkpoint_coarse_5000.pth" -s data/waymo/NOTA_dy32/training/016  --port 6017 --expname 'waymo' --configs 'arguments/nvs.py' --model_path ./NOTA_10