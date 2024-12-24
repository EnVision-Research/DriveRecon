import torch
import torch.nn.functional as F
import math
import numpy as np
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer ## 3D GS
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer ## 2D GS
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
from utils.point_utils import depth_to_normal

import gc
import copy
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
# from utils.point_utils import addpoint, combine_pointcloud, downsample_point_cloud_open3d, find_indices_in_A
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness
from scene.PointNet import UNet, ResnetBlock
from utils.loss_utils import l1_loss, ssim, l2_loss, compute_depth
from utils.image_utils import psnr

def axis_angle_to_rotation_matrix(axis, angle):

    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)  # 确保轴为单位向量
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s, 0.000],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s, 0.000],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C, 0.000],
        [0.000, 0.000, 0.000, 1.000]
    ])

    return R


def maps2all(allmap, pipe, viewpoint_camera):
    # additional regularizations
    render_alpha = allmap[1:2]
    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @
                     (viewpoint_camera.world_view_transform[:3, :3].T.to(device=render_normal.device))).permute(2, 0, 1)
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    # get depth distortion map
    render_dist = allmap[6:7]
    pipe.depth_ratio = 0
    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    return surf_depth, surf_normal, render_alpha, render_normal, render_depth_median, render_depth_expected


def pearson_similarity(pred, label):
    label[label < 1] = 512.0
    pred_mean = pred.mean(dim=[1, 2], keepdim=True)
    label_mean = label.mean(dim=[1, 2], keepdim=True)

    pred_centered = pred - pred_mean
    label_centered = label - label_mean

    numerator = (pred_centered * label_centered).sum(dim=[1, 2])
    denominator = torch.sqrt((pred_centered ** 2).sum(dim=[1, 2]) * (label_centered ** 2).sum(dim=[1, 2]))

    similarity = numerator / denominator
    return similarity


class Guassian_Adaptor(nn.Module):
    def __init__(self, out_num, num_samples, gaussian_scale_min, gaussian_scale_max, seg_num):
        super(Guassian_Adaptor, self).__init__()

        self.out_num = out_num
        self.num_samples = num_samples
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        self.seg_num = seg_num
        self.max_shift = 5


        # RGB
        self.rgb_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 2)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
            nn.Conv2d(int(self.out_num / 2), 3, kernel_size=1),
            nn.Tanh()  # Apply scaling in forward
        )

        # scale
        self.atri_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 4)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 4)),
            nn.Conv2d(int(self.out_num / 4), 7, kernel_size=1),
        )

        # depth
        self.depth_map_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 2)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
            nn.Conv2d(int(self.out_num / 2), self.num_samples, kernel_size=1),
        )

        # position
        self.position_shift_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 2)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
            nn.Conv2d(int(self.out_num / 2), 5, kernel_size=1),
            nn.Tanh()
        )

        # depth_reg
        self.depth_regression_head = nn.Sequential(
            ResnetBlock(self.out_num, int(self.out_num / 2)),
            # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
            nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
            nn.Conv2d(int(self.out_num / 2), 1, kernel_size=1),
            nn.Tanh()
        )

        # seg
        if self.seg_num > 0:
            self.seg_head = nn.Sequential(
                ResnetBlock(self.out_num, int(self.out_num / 2)),
                # ResnetBlock(int(self.out_num / 8), int(self.out_num / 8)),
                nn.GroupNorm(num_groups=4, num_channels=int(self.out_num / 2)),
                nn.Conv2d(int(self.out_num / 2), self.seg_num, kernel_size=1)
            )


    def forward(self, x):
        _, C, H, W = x.shape

        rgb_output = self.rgb_head(x)
        rgb_output = 0.5 * rgb_output + 0.5  # Scale to range [0, 1]

        atri_output = self.atri_head(x)
        scale_output = self.gaussian_scale_min + 0.1 * (
                    self.gaussian_scale_max - self.gaussian_scale_min) * F.softplus(atri_output[:, 0:3, ...])
        rotation_output = F.normalize(atri_output[:, 3:7, ...], dim=1)
        alpha_output = torch.sigmoid(atri_output[:, -1, ...])

        position_shift = self.position_shift_head(x)
        uv_shift = self.max_shift * position_shift[:, 0:2, ...]
        means_shift = self.max_shift * position_shift[:, 2:, ...]
        depth_map_output = self.depth_map_head(x)

        depth_regression_output = self.depth_regression_head(x)
        if self.seg_num > 0:
            seg_output = self.seg_head(x)
            seg_output = seg_output.permute(0, 2, 3, 1)
        else:
            seg_output = None

        rgb_output = rgb_output.permute(0, 2, 3, 1)
        scale_output = scale_output.permute(0, 2, 3, 1)
        rotation_output = rotation_output.permute(0, 2, 3, 1)
        depth_map_output = depth_map_output.permute(0, 2, 3, 1)
        depth_regression_output = depth_regression_output.permute(0, 2, 3, 1)

        return rgb_output, scale_output, alpha_output, rotation_output, \
               depth_map_output, depth_regression_output, seg_output, \
               uv_shift, means_shift



class Gaussian_LRM(nn.Module):
    def __init__(self, sh_degree: int, args, View_Num=3, shs_pre=True):
        super().__init__()
        self.shs_pre = shs_pre
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.View_Num = View_Num

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.count = 0
        self._deformation_table = torch.empty(0)
        self.setup_functions()
        self.super_Resolution = 1

        self.num_samples = 200
        gaussian_scale_max = 4
        gaussian_scale_min = 0.001

        # depth
        self.depth_min = 0.1
        self.depth_max = 400
        self.range = self.depth_min + torch.linspace(0.0, 1.0, self.num_samples) * (self.depth_max - self.depth_min)
        # seg
        self.seg_num = 3

        self.input_num = 3
        self.out_num = 128
        self.unet = UNet(
            in_channels = self.input_num,  # 需要加位置
            out_channels = self.out_num
        )
        self.Guassian_Adaptor = Guassian_Adaptor(out_num=self.out_num, num_samples=self.num_samples,
                                                 gaussian_scale_min=gaussian_scale_min,
                                                 gaussian_scale_max=gaussian_scale_max,
                                                 seg_num = self.seg_num)

        self.unet.to(torch.bfloat16).cuda()
        self.Guassian_Adaptor.to(torch.bfloat16).cuda()

    def forward(self, cams_input, images, depths, semantic_mask, intrinsics,
                c2ws, rendered_views, pipe, background, stage,
                sh_flag=True, LRM_flag=True, max_sh_degree=3):

        N, T, V, C, H, W = images.size()
        intrinsics = intrinsics.reshape(N*T*V, 3, 3)
        c2ws = c2ws.reshape(N*T*V, 4, 4)
        depth_Prob, scales, rotations, opacitys, rgbs_or_shs, depth_refined, seg_pre, posi_shift, means_shift, loss_pe = self.forward_gaussians(cams_input, intrinsics, c2ws, depths)

        depth_weight = 2
        seg_weight = 1
        depth_back_weight = 2
        depth_sky_weight = 0
        lambda_normal = 0
        static_weight = 0.1
        dy_weight = 0.1
        # seg约束
        if seg_pre is None:
            loss = 0
        else:
            semantic_mask = semantic_mask.reshape(N * T, -1).to(seg_pre.device)
            loss = seg_weight * F.cross_entropy(seg_pre.permute(0, 2, 1), semantic_mask.long())
            if torch.isnan(loss):
                print("pre seg_loss is NaN")
                loss = 0
        if torch.isnan(loss_pe):
            print("pre loss_pe is NaN")
        else:
            loss += depth_weight * loss_pe
        # 深度约束
        depths_gt = 255 * depths.reshape(N * T * V, H, W).to(dtype=depth_Prob.dtype).to(depth_Prob.device)



        Depth_pre = depth_Prob.reshape(N * T * V, H, W, self.num_samples)
        self.range = self.range.to(Depth_pre.device).reshape(1, 1, 1, self.num_samples)

        mask = depths_gt > 0.1
        depth_background_loss = F.cross_entropy(Depth_pre[mask], depths_gt[mask].long(), reduction='mean')
        if torch.isnan(depth_background_loss):
            pass
        else:
            loss += depth_back_weight * depth_background_loss

        semantic_mask = semantic_mask.reshape(N * T * V, H, W)


        Sky_mask = (semantic_mask == 2)
        Sky_depth = (torch.ones_like(depths_gt) * self.num_samples - 1).long()
        depth_sky_loss = F.cross_entropy(Depth_pre[Sky_mask], Sky_depth[Sky_mask].long(), reduction='mean')
        if torch.isnan(depth_sky_loss):
            pass
        else:
            loss += depth_sky_weight * depth_sky_loss

        Depth_pre = (Depth_pre.softmax(-1) * self.range).sum(dim=-1)
        depth_refined = depth_refined.reshape(N * T * V, H, W)
        Depth_pre = Depth_pre + depth_refined
        depths_loss_reg = compute_depth("l2", Depth_pre[mask], depths_gt[mask])
        if torch.isnan(depths_loss_reg):
            print("pre depths_loss_reg is NaN")
        else:
            loss += depth_weight * depths_loss_reg
        # 转换
        means3Ds = self.depth2xyz(Depth_pre, posi_shift, intrinsics, c2ws, N=N * T * V, H=H, W=W)
        means3Ds = means3Ds.reshape(N * T, -1, 3).float()

        if sh_flag:
            rgbs_or_shs = rgbs_or_shs.reshape(N*T, -1, (max_sh_degree + 1) ** 2, 3)
        else:
            rgbs_or_shs = rgbs_or_shs.reshape(N*T, -1, 3)

        screenspace_points = torch.zeros_like(means3Ds, dtype=torch.float32, requires_grad=True,
                                              device=means3Ds.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds = screenspace_points
        means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds = \
            torch.squeeze(means3Ds), torch.squeeze(scales), \
            torch.squeeze(rotations), opacitys.squeeze(0), \
            torch.squeeze(rgbs_or_shs), torch.squeeze(means2Ds)

        psnr_ = 0
        self.count = self.count + 1

        # the same time
        rendered_num = 0
        for index in range(int(len(rendered_views))):  # /V
            rendered_num = rendered_num + 1
            sub_time = int(index/V)
            rendered_view = rendered_views[index]
            means3D, scale, rotation, opacity, rgbs_or_sh, means2D = \
                torch.squeeze(means3Ds[sub_time]), torch.squeeze(scales[sub_time]), \
                torch.squeeze(rotations[sub_time]), opacitys[sub_time].squeeze(0), \
                torch.squeeze(rgbs_or_shs[sub_time]), torch.squeeze(means2Ds[sub_time])

            index_rendered = index % 3
            start_index = max(0, int(H * W * (index_rendered - 0.2)))
            start_end = min(int(H * W * (index_rendered + 1 + 0.2)), V * H * W)


            means3D_view, scale_view, rotation_view, \
            opacity_view, rgbs_or_sh_view, means2D_view = means3D[start_index:start_end], \
                                                         scale[start_index:start_end], \
                                                         rotation[start_index:start_end], \
                                                         opacity[start_index:start_end], \
                                                         rgbs_or_sh[start_index:start_end], \
                                                         means2D[start_index:start_end]

            time = torch.tensor(rendered_view.time).to(means3D_view.device).repeat(means3D_view.shape[1], 1)
            Guassian_para = {
                        'means3D': means3D_view,
                        'scale': scale_view,
                        'rotation': rotation_view,
                        'opacity': opacity_view,
                        'rgbs_or_shs': rgbs_or_sh_view,
                        'means2D': means2D_view,
                        'time': time,
                    }
            image_rendered, radii, depth_rendered, surf_normal, render_normal = self.gs_render(Guassian_para, rendered_view, pipe, background,
                                        stage=stage, return_dx=True,
                                        render_feat=True if ('fine' in stage and args.feat_head) else False,
                                        sh_flag=sh_flag, LRM_flag=LRM_flag)
            gt_image = rendered_view.original_image.to(device=means3Ds.device)
            normal_error = (1 - (render_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            Ll1 = l1_loss(image_rendered, gt_image)
            psnr_ += psnr(image_rendered, gt_image).mean().double()
            if torch.isnan(Ll1):
               pass
            else:
                loss = loss + Ll1
            if torch.isnan(normal_loss):
                pass
            else:
                loss = loss + normal_loss


        ## cross time for static
        for index_rendered in range(0, int(len(rendered_views))):
            sub_time = int(index_rendered / (T * V))
            rendered_view = rendered_views[index_rendered]
            scence_index = sub_time * T + randint(1, T - 1)

            means3D, scale, rotation, opacity, rgbs_or_sh, means2D = \
                torch.squeeze(means3Ds[scence_index]), torch.squeeze(scales[scence_index]), \
                torch.squeeze(rotations[scence_index]), opacitys[scence_index].squeeze(0), \
                torch.squeeze(rgbs_or_shs[scence_index]), torch.squeeze(means2Ds[scence_index])
            time = torch.tensor(rendered_view.time).to(means3D.device).repeat(means3D.shape[1], 1)
            Guassian_para = {
                        'means3D': means3D,
                        'scale': scale,
                        'rotation': rotation,
                        'opacity': opacity,
                        'rgbs_or_shs': rgbs_or_sh,
                        'means2D': means2D,
                        'time': time,
                    }
            image_rendered, radii, depth_rendered, _, _= self.gs_render(Guassian_para, rendered_view, pipe, background,
                                        stage=stage, return_dx=True,
                                        render_feat=True if ('fine' in stage and args.feat_head) else False,
                                        sh_flag=sh_flag, LRM_flag=LRM_flag)

            gt_image = rendered_view.original_image.to(device=means3Ds.device)

            semantic_mask = rendered_view.semantic_mask.to(device=means3Ds.device)
            static_mask = semantic_mask != 1
            static_mask_3 = static_mask.repeat(3, 1, 1)
            render_mask = image_rendered > 0.1
            Ll1 = l1_loss(image_rendered[static_mask_3 & render_mask], gt_image[static_mask_3 & render_mask])  # + xyz_loss/xyz_loss.item()
            if torch.isnan(Ll1):
                print("static Ll1 is NaN")
            else:
                loss += (static_weight * Ll1)/int(len(rendered_views)) # + rgb_loss


        ## cross time for dynamic
        for sub_view in range(0, N):
            time_index = randint(0, T-2)
            scence_index = sub_view * T + time_index
            index_rendered = (scence_index +1) * V + randint(0, V-1)
            rendered_view = rendered_views[index_rendered]


            means3D, scale, rotation, opacity, rgbs_or_sh, means2D = \
                torch.squeeze(means3Ds[scence_index]), torch.squeeze(scales[scence_index]), \
                torch.squeeze(rotations[scence_index]), opacitys[scence_index].squeeze(0), \
                torch.squeeze(rgbs_or_shs[scence_index]), torch.squeeze(means2Ds[scence_index])
            means3D = means3D + torch.squeeze(means_shift[scence_index])
            time = torch.tensor(rendered_view.time).to(means3D.device).repeat(means3D.shape[1], 1)
            Guassian_para = {
                        'means3D': means3D,
                        'scale': scale,
                        'rotation': rotation,
                        'opacity': opacity,
                        'rgbs_or_shs': rgbs_or_sh,
                        'means2D': means2D,
                        'time': time,
                    }
            image_rendered, radii, depth_rendered, _, _= self.gs_render(Guassian_para, rendered_view, pipe, background,
                                        stage=stage, return_dx=True,
                                        render_feat=True if ('fine' in stage and args.feat_head) else False,
                                        sh_flag=sh_flag, LRM_flag=LRM_flag)

            gt_image = rendered_view.original_image.to(device=means3Ds.device)
            # gt_depth = rendered_view.depth_map.to(device=means3Ds.device)

            semantic_mask = rendered_view.semantic_mask.to(device=means3Ds.device)
            static_mask = semantic_mask != 1
            static_mask_3 = static_mask.repeat(3, 1, 1)
            render_mask = image_rendered > 0.1
            Ll1 = l1_loss(image_rendered[static_mask_3 & render_mask], gt_image[static_mask_3 & render_mask])  # + xyz_loss/xyz_loss.item()
            if torch.isnan(Ll1):
                pass
            else:
                loss += (dy_weight * Ll1)/N

        return loss, psnr_/(rendered_num) #

    def transform_depth(self, d_raw, a=50, b=0.1, c=1):
        return a * torch.log10(b * d_raw + 1)

    def forward_gaussians(self, images, intrinsics, c2ws, depths):
        # B = 1
        # images: [3, 3, H, W] # 需要加位置
        # return: Gaussians: [B, dim_t]
        self.device = next(self.unet.conv_in.parameters()).device

        B, T, V, C, H, W = images.shape
        images = images.to(device=self.device)
        intrinsics = intrinsics.to(device=self.device)
        c2ws = c2ws.to(device=self.device)
        depths = depths.to(device=self.device)

        x, loss = self.unet(images.reshape(B * T * V, C, H, W),
                      intrinsics.reshape(B * T * V, 3, 3),
                      c2ws.reshape(B * T * V, 4, 4),
                      depths)

        rgbs_or_shs, scale, opacity, rotation, depth_prob, depth_refined, seg, posi_shift, means_shift = self.Guassian_Adaptor(x)
        depth_refined = depth_refined * (self.depth_max - self.depth_min) / self.num_samples
        depth_prob = depth_prob.reshape(B * T, -1, self.num_samples).contiguous().float()
        scale = scale.reshape(B * T, -1, 3).contiguous().float()
        rotation = rotation.reshape(B * T, -1, 4).contiguous().float()
        opacity = opacity.reshape(B * T, -1, 1).contiguous().float()
        rgbs_or_shs = rgbs_or_shs.reshape(B * T, -1, 3).contiguous().float()
        means_shift = means_shift.reshape(B * T, -1, 3).contiguous().float()
        depth_refined = depth_refined.reshape(B * T, -1, 1).contiguous().float()
        if seg is not None:
            seg = seg.reshape(B * T, -1, self.seg_num).contiguous().float()
        return depth_prob, scale, rotation, opacity, rgbs_or_shs, depth_refined, seg, posi_shift, means_shift, loss

    def gs_render_mul(self, Guassian_para, viewpoint_camera, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
           stage="coarse", return_decomposition=False, return_dx=False, render_feat=False, sh_flag=False, LRM_flag=True):

        # Guassain Para
        means3D = Guassian_para['means3D']
        scale = Guassian_para['scale']
        rotation = Guassian_para['rotation']
        opacity = Guassian_para['opacity']
        rgbs_or_shs = Guassian_para['rgbs_or_shs']
        means2D = Guassian_para['means2D']
        time = Guassian_para['time']

        # viewpoint_camera
        viewpoint_camera.camera_center = viewpoint_camera.camera_center.float().to(means3D.device)
        viewpoint_camera.world_view_transform = viewpoint_camera.world_view_transform.float().to(means3D.device)
        viewpoint_camera.full_proj_transform = viewpoint_camera.full_proj_transform.float().to(means3D.device)
        # GaussianRasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        image_all = []
        for iii in range(61):
            # 示例用法
            axis = [0, 0, 1]
            angle = np.deg2rad(-12 + 0.4 * iii)  # 45 度
            RRRR = axis_angle_to_rotation_matrix(axis, angle)
            RRRR = torch.tensor(RRRR, dtype=torch.float32).to(viewpoint_camera.world_view_transform.device)
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=RRRR@viewpoint_camera.world_view_transform,
                projmatrix=RRRR@viewpoint_camera.full_proj_transform,
                sh_degree=self.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            cov3D_precomp = None
            if pipe.compute_cov3D_python:
                cov3D_precomp = self.get_covariance(scaling_modifier)

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)


            if "coarse" in stage:
                means3D_final, scales_final, rotations_final, opacity_final, rgbs_or_shs_final = means3D, scale, rotation, opacity, rgbs_or_shs
            elif "fine" in stage:
                # time0 = get_time()
                # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point],
                #                                                                  rotations[deformation_point], opacity[deformation_point],
                #                                                                  time[deformation_point])
                means3D_final, scales_final, rotations_final, opacity_final, rgbs_or_shs_final, dx, feat, dshs = pc._deformation(
                    means3D, scale,
                    scale, opacity, rgbs_or_shs,
                    time)
            else:
                raise NotImplementedError


            colors_precomp = None
            if override_color is None:
                if pipe.convert_SHs_python:
                    if sh_flag:
                        if LRM_flag:
                            shs_view = rgbs_or_shs_final.transpose(-2, -1).reshape(-1, 3, (self.max_sh_degree + 1) ** 2)
                            dir_pp = (means3D - viewpoint_camera.camera_center.cuda().repeat(shs_view.shape[0], 1))
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                        else:
                            shs_view = rgbs_or_shs_final.transpose(-2, -1).reshape(-1, 3, (self.max_sh_degree + 1) ** 2)
                            dir_pp = (self.get_xyz - viewpoint_camera.camera_center.cuda().repeat(self.get_features.shape[0], 1))
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)

                        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                    else:
                        colors_precomp = torch.squeeze(rgbs_or_shs_final)
                else:
                    pass
            else:
                colors_precomp = override_color

            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            # time3 = get_time()
            if colors_precomp is not None:
                shs_final = None
            scales_final = scales_final[:, 0:2]
            rendered_image, radii, allmap = rasterizer( # , depth, rendered_alpha
                means3D=means3D_final,
                means2D=means2D,
                shs=shs_final,
                colors_precomp=colors_precomp,  # [N,3]
                opacities=opacity_final,
                scales=scales_final,
                rotations=rotations_final,
                cov3D_precomp=None)
            rendered_image = rendered_image.clamp(0, 1)
            surf_depth, surf_normal, render_alpha, render_normal, render_depth_median, render_depth_expected = maps2all(allmap, pipe, viewpoint_camera)
            image_all.append(rendered_image)

        image_shift = []
        viewpoint_camera.T[0] -= 1.2
        viewpoint_camera.update()
        for iii in range(61):
            # 示例用法
            viewpoint_camera.T[0] += 0.04
            viewpoint_camera.update()
            viewpoint_camera.camera_center = viewpoint_camera.camera_center.float().to(means3D.device)
            viewpoint_camera.world_view_transform = viewpoint_camera.world_view_transform.float().to(means3D.device)
            viewpoint_camera.full_proj_transform = viewpoint_camera.full_proj_transform.float().to(means3D.device)
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=self.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            cov3D_precomp = None
            if pipe.compute_cov3D_python:
                cov3D_precomp = self.get_covariance(scaling_modifier)

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            if "coarse" in stage:
                means3D_final, scales_final, rotations_final, opacity_final, rgbs_or_shs_final = means3D, scale, rotation, opacity, rgbs_or_shs
            elif "fine" in stage:
                # time0 = get_time()
                # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point],
                #                                                                  rotations[deformation_point], opacity[deformation_point],
                #                                                                  time[deformation_point])
                means3D_final, scales_final, rotations_final, opacity_final, rgbs_or_shs_final, dx, feat, dshs = pc._deformation(
                    means3D, scale,
                    scale, opacity, rgbs_or_shs,
                    time)
            else:
                raise NotImplementedError

            colors_precomp = None
            if override_color is None:
                if pipe.convert_SHs_python:
                    if sh_flag:
                        if LRM_flag:
                            shs_view = rgbs_or_shs_final.transpose(-2, -1).reshape(-1, 3, (self.max_sh_degree + 1) ** 2)
                            dir_pp = (means3D - viewpoint_camera.camera_center.cuda().repeat(shs_view.shape[0], 1))
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                        else:
                            shs_view = rgbs_or_shs_final.transpose(-2, -1).reshape(-1, 3, (self.max_sh_degree + 1) ** 2)
                            dir_pp = (self.get_xyz - viewpoint_camera.camera_center.cuda().repeat(
                                self.get_features.shape[0], 1))
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)

                        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                    else:
                        colors_precomp = torch.squeeze(rgbs_or_shs_final)
                else:
                    pass
            else:
                colors_precomp = override_color

            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            # time3 = get_time()
            if colors_precomp is not None:
                shs_final = None
            scales_final = scales_final[:, 0:2]
            rendered_image, radii, allmap = rasterizer(  # , depth, rendered_alpha
                means3D=means3D_final,
                means2D=means2D,
                shs=shs_final,
                colors_precomp=colors_precomp,  # [N,3]
                opacities=opacity_final,
                scales=scales_final,
                rotations=rotations_final,
                cov3D_precomp=None)
            rendered_image = rendered_image.clamp(0, 1)
            surf_depth, surf_normal, render_alpha, render_normal, render_depth_median, render_depth_expected = maps2all(
                allmap, pipe, viewpoint_camera)
            image_shift.append(rendered_image)




        return image_all, rendered_image, radii, surf_depth, surf_normal, render_normal, image_shift


    def infer_mul(self, cams_input, images, depths, semantic_mask, intrinsics, c2ws, rendered_views, pipe,
              background, stage,
              sh_flag=True, LRM_flag=True, max_sh_degree=3):
        # images 输入的image
        # intrinsics 输入图像的内参
        # c2ws 外参
        # viewpoint_cam 需要渲染的参数
        N, T, V, C, H, W = images.size()
        intrinsics = intrinsics.reshape(N * T * V, 3, 3)
        c2ws = c2ws.reshape(N * T * V, 4, 4)
        with torch.no_grad():
            depth_Prob, scales, rotations, opacitys, rgbs_or_shs, depth_refined, seg_pre, posi_shift, means_shift, loss_pe = self.forward_gaussians(
                cams_input, intrinsics, c2ws, depths)


        Depth_pre = depth_Prob.reshape(N * T * V, H, W, self.num_samples)
        self.range = self.range.to(Depth_pre.device).reshape(1, 1, 1, self.num_samples)



        Depth_pre = (Depth_pre.softmax(-1) * self.range).sum(dim=-1)
        depth_refined = depth_refined.reshape(N * T * V, H, W)
        Depth_pre = Depth_pre + depth_refined

        means3Ds = self.depth2xyz(Depth_pre, posi_shift, intrinsics, c2ws, N=N * T * V, H=H, W=W)
        means3Ds = means3Ds.reshape(N * T, -1, 3).float()

        if sh_flag:
            rgbs_or_shs = rgbs_or_shs.reshape(N * T, -1, (max_sh_degree + 1) ** 2, 3)
        else:
            rgbs_or_shs = rgbs_or_shs.reshape(N * T, -1, 3)

        screenspace_points = torch.zeros_like(means3Ds, dtype=torch.float32, requires_grad=True,
                                              device=means3Ds.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds = screenspace_points
        means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds = \
            torch.squeeze(means3Ds), torch.squeeze(scales), \
            torch.squeeze(rotations), opacitys.squeeze(0), \
            torch.squeeze(rgbs_or_shs), torch.squeeze(means2Ds)

        psnr_ = 0
        self.count = self.count + 1

        # 同场景
        image_gts, image_recstuctions, depth_recstuctions, surf_recstuctions, normal_recstuctions = [], [], [], [], []
        image_dy, image_static = [], []
        rendered_num = 0
        Guassian_paras = []
        for index in range(int(len(rendered_views))):  # /V
            rendered_num = rendered_num + 1
            sub_view = int(index / V)
            rendered_view = rendered_views[index]
            # cams_input = images.reshape(N*T, V, C, H, W)
            # print(cams_input[sub_view, index%V] - rendered_view.original_image)
            means3D, scale, rotation, opacity, rgbs_or_sh, means2D = \
                torch.squeeze(means3Ds[sub_view]), torch.squeeze(scales[sub_view]), \
                torch.squeeze(rotations[sub_view]), opacitys[sub_view].squeeze(0), \
                torch.squeeze(rgbs_or_shs[sub_view]), torch.squeeze(means2Ds[sub_view])
            seg_pre_sub = semantic_mask.reshape(N * T, -1)  ####????
            seg_pre_sub = torch.squeeze(seg_pre_sub[sub_view])

            index_rendered = index % 3
            if index_rendered !=1:
                continue
            start_index = 0 #:max(0, int(H * W * (index_rendered - 0.4)))
            start_end = -1 # min(int(H * W * (index_rendered + 1 + 0.4)), V * H * W)


            means3D_view, scale_view, rotation_view, \
            opacity_view, rgbs_or_sh_view, means2D_view = means3D[start_index:start_end], \
                                                          scale[start_index:start_end], \
                                                          rotation[start_index:start_end], \
                                                          opacity[start_index:start_end], \
                                                          rgbs_or_sh[start_index:start_end], \
                                                          means2D[start_index:start_end]

            seg_argmax = seg_pre_sub[start_index:start_end]
            static_mask = seg_argmax != 1
            dy_mask = ~static_mask

            time = torch.tensor(rendered_view.time).to(means3D_view.device).repeat(means3D_view.shape[1], 1)
            Guassian_para = {
                'means3D': means3D_view,
                'scale': scale_view,
                'rotation': rotation_view,
                'opacity': opacity_view,
                'rgbs_or_shs': rgbs_or_sh_view,
                'means2D': means2D_view,
                'time': time,
            }
            Guassian_paras.append(Guassian_para)
            Guassian_para_dy = {
                'means3D': means3D_view[dy_mask],
                'scale': scale_view[dy_mask],
                'rotation': rotation_view[dy_mask],
                'opacity': opacity_view[dy_mask],
                'rgbs_or_shs': rgbs_or_sh_view[dy_mask],
                'means2D': means2D_view[dy_mask],
                'time': time,
            }
            Guassian_para_static = {
                'means3D': means3D_view[static_mask],
                'scale': scale_view[static_mask],
                'rotation': rotation_view[static_mask],
                'opacity': opacity_view[static_mask],
                'rgbs_or_shs': rgbs_or_sh_view[static_mask],
                'means2D': means2D_view[static_mask],
                'time': time,
            }

            image_all, image_rendered, radii, depth_rendered, surf_normal, render_normal, image_shift = self.gs_render_mul(Guassian_para,
                                                                                               rendered_view,
                                                                                               pipe, background,
                                                                                               stage=stage,
                                                                                               return_dx=True,
                                                                                               render_feat=True if (
                                                                                                           'fine' in stage and args.feat_head) else False,
                                                                                               sh_flag=sh_flag,
                                                                                               LRM_flag=LRM_flag)
            gt_image = rendered_view.original_image.to(device=means3Ds.device)
            image_gts.append(gt_image)

        # means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds
        return_diction = {
            'image_all': image_all,
            "image_shift": image_shift,
            'image_gts': image_gts,
        }

        return return_diction

    def gs_render(self, Guassian_para, viewpoint_camera, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
           stage="coarse", return_decomposition=False, return_dx=False, render_feat=False, sh_flag=False, LRM_flag=True):


        # Guassain Para
        means3D = Guassian_para['means3D']
        scale = Guassian_para['scale']
        rotation = Guassian_para['rotation']
        opacity = Guassian_para['opacity']
        rgbs_or_shs = Guassian_para['rgbs_or_shs']
        means2D = Guassian_para['means2D']
        time = Guassian_para['time']

        # viewpoint_camera
        viewpoint_camera.camera_center = viewpoint_camera.camera_center.float().to(means3D.device)
        viewpoint_camera.world_view_transform = viewpoint_camera.world_view_transform.float().to(means3D.device)
        viewpoint_camera.full_proj_transform = viewpoint_camera.full_proj_transform.float().to(means3D.device)
        # GaussianRasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)


        if "coarse" in stage:
            means3D_final, scales_final, rotations_final, opacity_final, rgbs_or_shs_final = means3D, scale, rotation, opacity, rgbs_or_shs
        elif "fine" in stage:
            # time0 = get_time()
            # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point],
            #                                                                  rotations[deformation_point], opacity[deformation_point],
            #                                                                  time[deformation_point])
            means3D_final, scales_final, rotations_final, opacity_final, rgbs_or_shs_final, dx, feat, dshs = pc._deformation(
                means3D, scale,
                scale, opacity, rgbs_or_shs,
                time)
        else:
            raise NotImplementedError


        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                if sh_flag:
                    if LRM_flag:
                        shs_view = rgbs_or_shs_final.transpose(-2, -1).reshape(-1, 3, (self.max_sh_degree + 1) ** 2)
                        dir_pp = (means3D - viewpoint_camera.camera_center.cuda().repeat(shs_view.shape[0], 1))
                        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                    else:
                        shs_view = rgbs_or_shs_final.transpose(-2, -1).reshape(-1, 3, (self.max_sh_degree + 1) ** 2)
                        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.cuda().repeat(self.get_features.shape[0], 1))
                        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)

                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    colors_precomp = torch.squeeze(rgbs_or_shs_final)
            else:
                pass
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # time3 = get_time()
        if colors_precomp is not None:
            shs_final = None
        scales_final = scales_final[:, 0:2]
        rendered_image, radii, allmap = rasterizer( # , depth, rendered_alpha
            means3D=means3D_final,
            means2D=means2D,
            shs=shs_final,
            colors_precomp=colors_precomp,  # [N,3]
            opacities=opacity_final,
            scales=scales_final,
            rotations=rotations_final,
            cov3D_precomp=None)
        rendered_image = rendered_image.clamp(0, 1)
        surf_depth, surf_normal, render_alpha, render_normal, render_depth_median, render_depth_expected = maps2all(allmap, pipe, viewpoint_camera)
        return rendered_image, radii, surf_depth, surf_normal, render_normal



    def free_render(self, Guassian_para,
                    viewpoint_camera, pipe,
                    bg_color: torch.Tensor,
                    scaling_modifier=1.0, override_color=None,
                    stage="coarse", return_decomposition=False,
                    return_dx=False, render_feat=False,
                    sh_flag=False, LRM_flag=True):


        # Guassain Para
        means3D = Guassian_para['means3D']
        scale = Guassian_para['scale']
        rotation = Guassian_para['rotation']
        opacity = Guassian_para['opacity']
        rgbs_or_shs = Guassian_para['rgbs_or_shs']
        means2D = Guassian_para['means2D']
        time = Guassian_para['time']

        # viewpoint_camera
        viewpoint_camera.camera_center = viewpoint_camera.camera_center.float().to(means3D.device)
        viewpoint_camera.world_view_transform = viewpoint_camera.world_view_transform.float().to(means3D.device)
        viewpoint_camera.full_proj_transform = viewpoint_camera.full_proj_transform.float().to(means3D.device)
        # GaussianRasterization
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)


        if "coarse" in stage:
            means3D_final, scales_final, rotations_final, opacity_final, rgbs_or_shs_final = means3D, scale, rotation, opacity, rgbs_or_shs
        elif "fine" in stage:
            # time0 = get_time()
            # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point],
            #                                                                  rotations[deformation_point], opacity[deformation_point],
            #                                                                  time[deformation_point])
            means3D_final, scales_final, rotations_final, opacity_final, rgbs_or_shs_final, dx, feat, dshs = pc._deformation(
                means3D, scale,
                scale, opacity, rgbs_or_shs,
                time)
        else:
            raise NotImplementedError


        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                if sh_flag:
                    if LRM_flag:
                        shs_view = rgbs_or_shs_final.transpose(-2, -1).reshape(-1, 3, (self.max_sh_degree + 1) ** 2)
                        dir_pp = (means3D - viewpoint_camera.camera_center.cuda().repeat(shs_view.shape[0], 1))
                        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                    else:
                        shs_view = rgbs_or_shs_final.transpose(-2, -1).reshape(-1, 3, (self.max_sh_degree + 1) ** 2)
                        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.cuda().repeat(self.get_features.shape[0], 1))
                        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)

                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    colors_precomp = torch.squeeze(rgbs_or_shs_final)
            else:
                pass
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # time3 = get_time()
        if colors_precomp is not None:
            shs_final = None
        scales_final = scales_final[:, 0:2]
        rendered_image, radii, allmap = rasterizer( # , depth, rendered_alpha
            means3D=means3D_final,
            means2D=means2D,
            shs=shs_final,
            colors_precomp=colors_precomp,  # [N,3]
            opacities=opacity_final,
            scales=scales_final,
            rotations=rotations_final,
            cov3D_precomp=None)
        rendered_image = rendered_image.clamp(0, 1)
        surf_depth, surf_normal, render_alpha, render_normal, render_depth_median, render_depth_expected = maps2all(allmap, pipe, viewpoint_camera)
        return rendered_image, radii, surf_depth, surf_normal, render_normal

    def infer(self, cams_input, images, depths, semantic_mask, intrinsics, c2ws, rendered_views, pipe, background, stage,
                sh_flag=True, LRM_flag=True, max_sh_degree=3):

        N, T, V, C, H, W = images.size()
        intrinsics = intrinsics.reshape(N * T * V, 3, 3)
        c2ws = c2ws.reshape(N * T * V, 4, 4)
        with torch.no_grad():
            depth_Prob, scales, rotations, opacitys, rgbs_or_shs, depth_refined, seg_pre, posi_shift, means_shift, loss_pe = self.forward_gaussians(
                cams_input, intrinsics, c2ws, depths)


        Depth_pre = depth_Prob.reshape(N * T * V, H, W, self.num_samples)
        self.range = self.range.to(Depth_pre.device).reshape(1, 1, 1, self.num_samples)



        Depth_pre = (Depth_pre.softmax(-1) * self.range).sum(dim=-1)
        depth_refined = depth_refined.reshape(N * T * V, H, W)
        Depth_pre = Depth_pre + depth_refined

        means3Ds = self.depth2xyz(Depth_pre, posi_shift, intrinsics, c2ws, N=N * T * V, H=H, W=W)
        means3Ds = means3Ds.reshape(N * T, -1, 3).float()

        if sh_flag:
            rgbs_or_shs = rgbs_or_shs.reshape(N * T, -1, (max_sh_degree + 1) ** 2, 3)
        else:
            rgbs_or_shs = rgbs_or_shs.reshape(N * T, -1, 3)

        screenspace_points = torch.zeros_like(means3Ds, dtype=torch.float32, requires_grad=True,
                                              device=means3Ds.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds = screenspace_points
        means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds = \
            torch.squeeze(means3Ds), torch.squeeze(scales), \
            torch.squeeze(rotations), opacitys.squeeze(0), \
            torch.squeeze(rgbs_or_shs), torch.squeeze(means2Ds)

        psnr_ = 0
        self.count = self.count + 1

        image_gts, image_recstuctions, depth_recstuctions, surf_recstuctions, normal_recstuctions = [], [], [], [], []
        image_dy, image_static = [], []
        rendered_num = 0
        Guassian_paras = []
        for index in range(int(len(rendered_views))):  # /V
            rendered_num = rendered_num + 1
            sub_view = int(index / V)
            rendered_view = rendered_views[index]
            # cams_input = images.reshape(N*T, V, C, H, W)
            # print(cams_input[sub_view, index%V] - rendered_view.original_image)
            means3D, scale, rotation, opacity, rgbs_or_sh, means2D = \
                torch.squeeze(means3Ds[sub_view]), torch.squeeze(scales[sub_view]), \
                torch.squeeze(rotations[sub_view]), opacitys[sub_view].squeeze(0), \
                torch.squeeze(rgbs_or_shs[sub_view]), torch.squeeze(means2Ds[sub_view])
            seg_pre_sub = torch.squeeze(seg_pre[sub_view])
            seg_pre_sub = semantic_mask.reshape(N * T, -1)  # To better evaluate dynamic and static objects, here we use ground-truth to replace the predicted,
            seg_pre_sub = torch.squeeze(seg_pre_sub[sub_view])

            index_rendered = index % 3
            start_index = max(0, int(H * W * (index_rendered - 0.0)))
            start_end = min(int(H * W * (index_rendered + 1 + 0.0)), V * H * W)

            means3D_view, scale_view, rotation_view, \
            opacity_view, rgbs_or_sh_view, means2D_view = means3D[start_index:start_end], \
                                                          scale[start_index:start_end], \
                                                          rotation[start_index:start_end], \
                                                          opacity[start_index:start_end], \
                                                          rgbs_or_sh[start_index:start_end], \
                                                          means2D[start_index:start_end]
            # seg_pre_sub_view = seg_pre_sub[start_index:start_end]
            # seg_argmax = seg_pre_sub_view.argmax(dim=-1) ????
            seg_argmax = seg_pre_sub[start_index:start_end]
            static_mask = seg_argmax != 1
            dy_mask = ~static_mask

            time = torch.tensor(rendered_view.time).to(means3D_view.device).repeat(means3D_view.shape[1], 1)
            Guassian_para = {
                'means3D': means3D_view,
                'scale': scale_view,
                'rotation': rotation_view,
                'opacity': opacity_view,
                'rgbs_or_shs': rgbs_or_sh_view,
                'means2D': means2D_view,
                'time': time,
            }
            Guassian_paras.append(Guassian_para)
            Guassian_para_dy = {
                'means3D': means3D_view[dy_mask],
                'scale': scale_view[dy_mask],
                'rotation': rotation_view[dy_mask],
                'opacity': opacity_view[dy_mask],
                'rgbs_or_shs': rgbs_or_sh_view[dy_mask],
                'means2D': means2D_view[dy_mask],
                'time': time,
            }
            Guassian_para_static = {
                'means3D': means3D_view[static_mask],
                'scale': scale_view[static_mask],
                'rotation': rotation_view[static_mask],
                'opacity': opacity_view[static_mask],
                'rgbs_or_shs': rgbs_or_sh_view[static_mask],
                'means2D': means2D_view[static_mask],
                'time': time,
            }

            image_rendered, radii, depth_rendered, surf_normal, render_normal = self.gs_render(Guassian_para, rendered_view,
                                                                                        pipe, background,
                                                                                        stage=stage, return_dx=True,
                                                                                        render_feat=True if ('fine' in stage and args.feat_head) else False,
                                                                                        sh_flag=sh_flag,
                                                                                        LRM_flag=LRM_flag)
            image_rendered_dy, _, _, _, _ = self.gs_render(Guassian_para_dy, rendered_view,
                                                                            pipe, background,
                                                                            stage=stage, return_dx=True,
                                                                            render_feat=True if ('fine' in stage and args.feat_head) else False,
                                                                            sh_flag=sh_flag,
                                                                            LRM_flag=LRM_flag)
            image_rendered_static, _, _, _, _ = self.gs_render(Guassian_para_static, rendered_view,
                                                                            pipe, background,
                                                                            stage=stage, return_dx=True,
                                                                            render_feat=True if ('fine' in stage and args.feat_head) else False,
                                                                            sh_flag=sh_flag,
                                                                            LRM_flag=LRM_flag)
            gt_image = rendered_view.original_image.to(device=means3Ds.device)
            image_gts.append(gt_image.detach().cpu())
            image_recstuctions.append(image_rendered.detach().cpu())
            depth_recstuctions.append(depth_rendered.detach().cpu())
            surf_recstuctions.append(surf_normal.detach().cpu())
            normal_recstuctions.append(render_normal.detach().cpu())
            image_dy.append(image_rendered_dy.detach().cpu())
            image_static.append(image_rendered_static.detach().cpu())


        # means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds
        return_diction = {
                'image_gts': image_gts,
                "Guassian_paras": Guassian_paras,
                "image_dy": image_dy,
                "image_static": image_static,
                'rend_alpha': radii,
                'image_recstuctions': image_recstuctions,
                'depth_recstuctions': Depth_pre,
                'surf_recstuctions': surf_recstuctions,
                'normal_recstuctions': normal_recstuctions,
                'semantic_mask': semantic_mask.reshape(N*T*V, H, W),
                'semantic_pre': seg_pre.reshape(N*T*V, H, W, 3),
        }

        return return_diction

    def noval_view(self, cams_input, images,  depths, semantic_mask, intrinsics, c2ws, rendered_views, pipe, background, stage,
                sh_flag=True, LRM_flag=True, max_sh_degree=3):

        N, T, V, C, H, W = images.size()
        intrinsics = intrinsics.reshape(N * T * V, 3, 3)
        c2ws = c2ws.reshape(N * T * V, 4, 4)
        with torch.no_grad():
            depth_Prob, scales, rotations, \
            opacitys, rgbs_or_shs, depth_refined,\
            seg_pre, posi_shift, means_shift, loss_pe = self.forward_gaussians(
                cams_input, intrinsics, c2ws, depths)



        Depth_pre = depth_Prob.reshape(N * T * V, H, W, self.num_samples)
        self.range = self.range.to(Depth_pre.device).reshape(1, 1, 1, self.num_samples)



        Depth_pre = (Depth_pre.softmax(-1) * self.range).sum(dim=-1)
        depth_refined = depth_refined.reshape(N * T * V, H, W)
        Depth_pre = Depth_pre + depth_refined
        means3Ds = self.depth2xyz(Depth_pre, posi_shift, intrinsics, c2ws, N=N * T * V, H=H, W=W)
        means3Ds = means3Ds.reshape(N * T, -1, 3).float()

        if sh_flag:
            rgbs_or_shs = rgbs_or_shs.reshape(N * T, -1, (max_sh_degree + 1) ** 2, 3)
        else:
            rgbs_or_shs = rgbs_or_shs.reshape(N * T, -1, 3)

        screenspace_points = torch.zeros_like(means3Ds, dtype=torch.float32, requires_grad=True,
                                              device=means3Ds.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds = screenspace_points
        means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds = \
            torch.squeeze(means3Ds), torch.squeeze(scales), \
            torch.squeeze(rotations), opacitys.squeeze(0), \
            torch.squeeze(rgbs_or_shs), torch.squeeze(means2Ds)

        psnr_ = 0
        self.count = self.count + 1

        # 新视角
        image_gts, image_recstuctions, depth_recstuctions, surf_recstuctions, normal_recstuctions = [], [], [], [], []
        rendered_num = 0
        for index in range(int(len(rendered_views))):  # /V
            rendered_num = rendered_num + 1
            sub_view = 1
            rendered_view = rendered_views[index]
            # cams_input = images.reshape(N*T, V, C, H, W)
            # print(cams_input[sub_view, index%V] - rendered_view.original_image)
            means3D, scale, rotation, opacity, rgbs_or_sh, means2D = \
                torch.squeeze(means3Ds[sub_view]), torch.squeeze(scales[sub_view]), \
                torch.squeeze(rotations[sub_view]), opacitys[sub_view].squeeze(0), \
                torch.squeeze(rgbs_or_shs[sub_view]), torch.squeeze(means2Ds[sub_view])
            # time = torch.tensor(rendered_view.time).to(means3D.device).repeat(means3D.shape[1], 1)
            # Guassian_para = {
            #     'means3D': means3D,
            #     'scale': scale,
            #     'rotation': rotation,
            #     'opacity': opacity,
            #     'rgbs_or_shs': rgbs_or_sh,
            #     'means2D': means2D,
            #     'time': time,
            # }
            index_rendered = index % 3
            start_index = max(0, int(H * W * (index_rendered - 0.2)))
            start_end = min(int(H * W * (index_rendered + 1 + 0.2)), V * H * W)

            # means3D_view, scale_view, rotation_view, \
            # opacity_view, rgbs_or_sh_view, means2D_view = means3D.reshape(V, H*W, -1), \
            #                                              scale.reshape(V, H*W, -1),\
            #                                              rotation.reshape(V, H*W, -1), \
            #                                              opacity.reshape(V, H*W, -1), \
            #                                              rgbs_or_sh.reshape(V, H*W, -1), \
            #                                              means2D.reshape(V, H*W, -1)
            means3D_view, scale_view, rotation_view, \
            opacity_view, rgbs_or_sh_view, means2D_view = means3D[start_index:start_end], \
                                                          scale[start_index:start_end], \
                                                          rotation[start_index:start_end], \
                                                          opacity[start_index:start_end], \
                                                          rgbs_or_sh[start_index:start_end], \
                                                          means2D[start_index:start_end]

            time = torch.tensor(rendered_view.time).to(means3D_view.device).repeat(means3D_view.shape[1], 1)
            Guassian_para = {
                'means3D': means3D_view,
                'scale': scale_view,
                'rotation': rotation_view,
                'opacity': opacity_view,
                'rgbs_or_shs': rgbs_or_sh_view,
                'means2D': means2D_view,
                'time': time,
            }

            image_rendered, radii, depth_rendered, surf_normal, render_normal = self.free_render(Guassian_para, rendered_view,
                                                                                        pipe, background,
                                                                                        stage=stage, return_dx=True,
                                                                                        render_feat=True if (
                                                                                                    'fine' in stage and args.feat_head) else False,
                                                                                        sh_flag=sh_flag,
                                                                                        LRM_flag=LRM_flag)
            gt_image = rendered_view.original_image.to(device=means3Ds.device)
            image_gts.append(gt_image.detach().cpu())
            image_recstuctions.append(image_rendered.detach().cpu())
            depth_recstuctions.append(depth_rendered.detach().cpu())
            surf_recstuctions.append(surf_normal.detach().cpu())
            normal_recstuctions.append(render_normal.detach().cpu())
        # means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds
        return_diction = {
                'image_gts': image_gts,
                'image_recstuctions': image_recstuctions,
                'depth_recstuctions': depth_recstuctions,
                'surf_recstuctions': surf_recstuctions,
                'normal_recstuctions': normal_recstuctions,
                'semantic_mask': semantic_mask.reshape(N*T*V, H, W),
                'semantic_pre': seg_pre.reshape(N*T*V, H, W, 3),
        }

        return return_diction

    def free_view(self, cams_input, images, depths, semantic_mask, intrinsics, c2ws, rendered_views, pipe, background, stage,
                sh_flag=True, LRM_flag=True, max_sh_degree=3):

        N, T, V, C, H, W = images.size()
        intrinsics = intrinsics.reshape(N * T * V, 3, 3)
        c2ws = c2ws.reshape(N * T * V, 4, 4)
        with torch.no_grad():
            depth_Prob, scales, rotations, opacitys, rgbs_or_shs, depth_refined, seg_pre, posi_shift, means_shift, loss_pe = self.forward_gaussians(
                cams_input, intrinsics, c2ws, depths)



        Depth_pre = depth_Prob.reshape(N * T * V, H, W, self.num_samples)
        self.range = self.range.to(Depth_pre.device).reshape(1, 1, 1, self.num_samples)



        Depth_pre = (Depth_pre.softmax(-1) * self.range).sum(dim=-1)
        depth_refined = depth_refined.reshape(N * T * V, H, W)
        Depth_pre = Depth_pre + depth_refined

        # 转换
        means3Ds = self.depth2xyz(Depth_pre, posi_shift, intrinsics, c2ws, N=N * T * V, H=H, W=W)
        means3Ds = means3Ds.reshape(N * T, -1, 3).float()

        if sh_flag:
            rgbs_or_shs = rgbs_or_shs.reshape(N * T, -1, (max_sh_degree + 1) ** 2, 3)
        else:
            rgbs_or_shs = rgbs_or_shs.reshape(N * T, -1, 3)

        screenspace_points = torch.zeros_like(means3Ds, dtype=torch.float32, requires_grad=True,
                                              device=means3Ds.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds = screenspace_points
        means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds = \
            torch.squeeze(means3Ds), torch.squeeze(scales), \
            torch.squeeze(rotations), opacitys.squeeze(0), \
            torch.squeeze(rgbs_or_shs), torch.squeeze(means2Ds)

        psnr_ = 0
        self.count = self.count + 1

        # 新视角
        image_gts, image_recstuctions, depth_recstuctions, surf_recstuctions, normal_recstuctions = [], [], [], [], []
        rendered_num = 0
        for index in range(int(len(rendered_views))):  # /V
            rendered_num = rendered_num + 1
            sub_view = 1
            rendered_view = rendered_views[index]
            # cams_input = images.reshape(N*T, V, C, H, W)
            # print(cams_input[sub_view, index%V] - rendered_view.original_image)
            means3D, scale, rotation, opacity, rgbs_or_sh, means2D = \
                torch.squeeze(means3Ds[sub_view]), torch.squeeze(scales[sub_view]), \
                torch.squeeze(rotations[sub_view]), opacitys[sub_view].squeeze(0), \
                torch.squeeze(rgbs_or_shs[sub_view]), torch.squeeze(means2Ds[sub_view])
            # time = torch.tensor(rendered_view.time).to(means3D.device).repeat(means3D.shape[1], 1)
            # Guassian_para = {
            #     'means3D': means3D,
            #     'scale': scale,
            #     'rotation': rotation,
            #     'opacity': opacity,
            #     'rgbs_or_shs': rgbs_or_sh,
            #     'means2D': means2D,
            #     'time': time,
            # }
            index_rendered = index % 3
            start_index = max(0, int(H * W * (index_rendered - 0.2)))
            start_end = min(int(H * W * (index_rendered + 1 + 0.2)), V * H * W)

            # means3D_view, scale_view, rotation_view, \
            # opacity_view, rgbs_or_sh_view, means2D_view = means3D.reshape(V, H*W, -1), \
            #                                              scale.reshape(V, H*W, -1),\
            #                                              rotation.reshape(V, H*W, -1), \
            #                                              opacity.reshape(V, H*W, -1), \
            #                                              rgbs_or_sh.reshape(V, H*W, -1), \
            #                                              means2D.reshape(V, H*W, -1)
            means3D_view, scale_view, rotation_view, \
            opacity_view, rgbs_or_sh_view, means2D_view = means3D[start_index:start_end], \
                                                          scale[start_index:start_end], \
                                                          rotation[start_index:start_end], \
                                                          opacity[start_index:start_end], \
                                                          rgbs_or_sh[start_index:start_end], \
                                                          means2D[start_index:start_end]

            time = torch.tensor(rendered_view.time).to(means3D_view.device).repeat(means3D_view.shape[1], 1)
            Guassian_para = {
                'means3D': means3D_view,
                'scale': scale_view,
                'rotation': rotation_view,
                'opacity': opacity_view,
                'rgbs_or_shs': rgbs_or_sh_view,
                'means2D': means2D_view,
                'time': time,
            }

            image_rendered, radii, depth_rendered, surf_normal, render_normal = self.gs_render(Guassian_para, rendered_view,
                                                                                        pipe, background,
                                                                                        stage=stage, return_dx=True,
                                                                                        render_feat=True if (
                                                                                                    'fine' in stage and args.feat_head) else False,
                                                                                        sh_flag=sh_flag,
                                                                                        LRM_flag=LRM_flag)
            gt_image = rendered_view.original_image.to(device=means3Ds.device)
            image_gts.append(gt_image)
            image_recstuctions.append(image_rendered)
            depth_recstuctions.append(depth_rendered)
            surf_recstuctions.append(surf_normal)
            normal_recstuctions.append(render_normal)
        # means3Ds, scales, rotations, opacitys, rgbs_or_shs, means2Ds
        return_diction = {
                'image_gts': image_gts,
                'image_recstuctions': image_recstuctions,
                'depth_recstuctions': depth_recstuctions,
                'surf_recstuctions': surf_recstuctions,
                'normal_recstuctions': normal_recstuctions,
                'semantic_mask': semantic_mask.reshape(N*T*V, H, W),
                'semantic_pre': seg_pre.reshape(N*T*V, H, W, 3),
        }

        return return_diction


    def depth2xyz(self, Depth, posi_shift, intrinsics, c2ws, N=3, H=640, W=960, scale=1):

        channel_1 = torch.arange(W).unsqueeze(0).expand(H, -1).repeat(N, 1, 1).to(Depth.device)
        channel_2 = torch.arange(H).unsqueeze(0).expand(W, -1).permute(1, 0).repeat(N, 1, 1).to(Depth.device)
        uv_map = torch.stack((channel_1, channel_2, Depth), dim=-1).to(torch.bfloat16)  # 三张图，H， W， u， v， depth
        posi_shift = posi_shift.permute(0, 2, 3, 1)
        uv_map[..., 0:2] += posi_shift
        cam_map = torch.zeros_like(uv_map).to(torch.bfloat16)
        cam_map[..., 2] += uv_map[..., 2]
        cam_map[..., 0:2] += torch.mul(uv_map[..., 0:2], uv_map[..., 2].unsqueeze(-1))
        cam_map = cam_map.reshape(N, -1, 3)
        xyzs = torch.zeros_like(cam_map).to(torch.bfloat16)
        xyzs_world = torch.zeros_like(cam_map).to(torch.bfloat16)
        for index in range(N):
            intrinsic, c2w = intrinsics[index], c2ws[index]
            xyzs[index,] += torch.mm(torch.inverse(intrinsic.to(cam_map.device).float()).to(torch.bfloat16), cam_map[index,].T).T
            temp_one = torch.ones_like(xyzs[index, :, 0]).unsqueeze(-1)
            temp = torch.cat((xyzs[index], temp_one), dim=-1)
            temp = torch.mm(c2w.to(uv_map.device), temp.T)
            xyzs_world[index] += temp[0:3, :].T
        xyzs = xyzs_world.reshape(N, H, W, 3)

        return xyzs


    def capture(self):
        return (
            self.unet.state_dict(),
            self.Guassian_Adaptor.state_dict(),
            self.optimizer.state_dict(),
        )

    def restore(self, model_args, training_args):
        (unet_state_dict,
         conv_state_dict,
         opt_dict) = model_args

        self.unet.load_state_dict(unet_state_dict)
        self.Guassian_Adaptor.load_state_dict(conv_state_dict)
        # self.optimizer.load_state_dict(opt_dict)

    def setup_functions(self):
        # def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        #     L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        #     actual_covariance = L @ L.transpose(1, 2)
        #     symm = strip_symmetric(actual_covariance)
        #     return symm
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                                        rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def oneupSHdegree(self):
        pass

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        l = [
            {'params': list(self.unet.parameters()), 'lr': 0.0004, 'weight_decay': 0.05, "name": "unet"},
            {'params': list(self.Guassian_Adaptor.parameters()), 'lr': 0.0004, 'weight_decay': 0.05, "name": "conv"},
        ]


        self.optimizer = torch.optim.Adam(l, lr=0.001, eps=1e-15)

        self.lrm_scheduler_args = get_expon_lr_func(lr_init=0.0004,
                                                     lr_final=0.001,
                                                     lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                     max_steps=training_args.position_lr_max_steps)
