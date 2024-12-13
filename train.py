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
from accelerate import Accelerator, DistributedDataParallelKwargs

import numpy as np
import time
import json
from utils.video_utils import render_pixels, save_videos
from utils.visualization_tools import compute_optical_flow_and_save
from scene.gaussian_model import merge_models
from scipy.interpolate import CubicSpline
from scene.dataset import WaymoDataset
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)
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

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname):
    ## Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1
    )
    stage = 'coarse'
    # model
    gaussians = Gaussian_LRM(dataset.sh_degree, hyper, View_Num=3, shs_pre=dataset.shs_pre)
    dataset.model_path = args.model_path
    train_iter = opt.coarse_iterations


    eval_dir = os.path.join(args.model_path, "eval")
    time_length = args.time_lenth
    View_Num = args.veiw_num
    rendered_num = args.rendered_num
    os.makedirs(eval_dir, exist_ok=True)
    file_root = "./data/waymo/NOTA2_64/training"
    _WaymoDataset = WaymoDataset(file_root, dataset, gaussians, time_lenth=args.time_lenth, load_coarse=None, Train_flag=True)
    batch_size = 1
    train_dataloader = torch.utils.data.DataLoader(
        _WaymoDataset,
        batch_size=batch_size,
        shuffle=True,
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

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    first_iter = 0
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    iteration_real = 0
    for iteration in range(first_iter, final_iter + 1):

        for i, scenes in enumerate(train_dataloader):
            with accelerator.accumulate(gaussians):
                # 读取多个场景流的多个视角
                viewpoint_stack = []
                for scene in scenes: # batchsize个场景
                    train_cams = scene.getTrainCameras()
                    for train_cam in train_cams:
                        viewpoint_stack.append(train_cam)

                cams = []
                cams_input = []
                xyzs = []
                depths = []
                depth_anys = []
                intrinsics = []
                c2ws = []
                semantic_mask = []
                for view_index in range(len(viewpoint_stack)):
                    cam = viewpoint_stack[view_index]
                    cams.append(cam.original_image)
                    cams_input.append(cam.image_input)
                    xyzs.append(cam.xyz_map)
                    depths.append(cam.depth_map)
                    semantic_mask.append(cam.semantic_mask)
                    intrinsics.append(cam.intrinsic)
                    c2ws.append(cam.c2w)
                # 把batch size, Tmeper放到前面来
                cams = torch.stack(cams, dim=0).reshape(batch_size, time_length, View_Num, 3, cams[0].shape[-2],
                                                        cams[0].shape[-1]).to(torch.bfloat16)
                semantic_mask = torch.stack(semantic_mask, dim=0).reshape(batch_size, time_length, View_Num, cams[0].shape[-2],
                                                        cams[0].shape[-1]).to(torch.bfloat16)
                cams_input = torch.stack(cams_input, dim=0).reshape(batch_size, time_length, View_Num, 3,
                                                                    cams_input[0].shape[-2],
                                                                    cams_input[0].shape[-1]).to(torch.bfloat16)
                xyzs = torch.stack(xyzs, dim=0).reshape(batch_size, time_length, View_Num, 3, xyzs[0].shape[-2],
                                                        xyzs[0].shape[-1]).to(torch.bfloat16)
                depths = torch.stack(depths, dim=0).reshape(batch_size, time_length, View_Num, 1, xyzs[0].shape[-2],
                                                            xyzs[0].shape[-1]).to(torch.bfloat16)

                intrinsics = torch.stack(intrinsics, dim=0).reshape(batch_size, time_length, View_Num, 3, 3).to(torch.bfloat16)  #
                c2ws = torch.stack(c2ws, dim=0).reshape(batch_size, time_length, View_Num, 4, 4).to(torch.bfloat16)
                # cams[0, 2, 2]-viewpoint_stack[4*3*0+3*2+2].original_image
                # breakpoint()

                per_scence_num = 500
                for temp_index in range(per_scence_num):
                    loss, psnr_ = gaussians(cams_input, cams, depths, semantic_mask,
                                            intrinsics, c2ws, viewpoint_stack, pipe=pipe,
                                            background=background, stage=stage,
                                           sh_flag=dataset.shs_pre)
                    if torch.isnan(loss).any():
                        print("loss is nan,end training, reexecv program now.")
                        loss = loss * 0.0
                        # os.execv(sys.executable, [sys.executable] + sys.argv)
                    accelerator.backward(loss)
                    iteration_real = iteration_real + 1


                    iter_end.record()

                    if iteration < final_iter + 1:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad()

                    with torch.no_grad():
                        # Progress bar
                        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                        ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log

                        if iteration % 1 == 0:
                            print_dict = {
                                "step": f"{iteration}",
                                "Loss": f"{ema_loss_for_log:.{7}f}",
                                "psnr": f"{psnr_:.{2}f}",
                            }
                            progress_bar.set_postfix(print_dict)
                            metrics_file = f"{scene.model_path}/logger.json"
                            with open(metrics_file, "a") as f:
                                json.dump(print_dict, f)
                                f.write('\n')
                            progress_bar.update(1)
                        if iteration == final_iter:
                            progress_bar.close()

                        # Log and save
                        timer.pause()

                        timer.start()
                if accelerator.is_local_main_process:
                    if (iteration_real in checkpoint_iterations):
                        save_path = "checkpoint_" + str(iteration_real) + ".pth"
                        for file in os.listdir(scene.model_path):
                            if file.endswith(".pth") and file != save_path:
                                os.remove(os.path.join(scene.model_path, file))


                        unwrap_gaussians = accelerator.unwrap_model(gaussians)
                        accelerator.save(unwrap_gaussians.capture(), save_path)
                        print(f"save model at " + save_path)










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
