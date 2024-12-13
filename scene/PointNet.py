import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import numpy as np
from typing import Tuple, Literal
from functools import partial

from scene.attention import MemEffAttention
from scene.PD_Block import PD_Block


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def basic_blocks(dim, out_dim, layers,
                 geo_flag=False,
                 mlp_ratio=1.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2,
                 heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(layers):
        blocks.append(PD_Block(
            dim, out_dim, geo_flag=geo_flag, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            proposal_w=proposal_w, proposal_h=proposal_h,
            fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim, return_center=return_center
        ))
        dim = out_dim
    blocks = nn.Sequential(*blocks)

    return blocks

class TCAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 3, # WARN: hardcoded!
        view_num: int = 3, # WARN: hardcoded!
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames
        self.view_num = view_num

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        # x: [B*V, C, H, W]
        BTV, C, H, W = x.shape
        B = BTV // (self.num_frames * self.view_num)# assert BV % self.num_frames == 0

        res = x
        x = self.norm(x)

        x = x.reshape(B, self.num_frames, self.view_num, C, H, W).permute(0, 1, 2, 4, 5, 3).reshape(B, -1, C)
        x = self.attn(x)
        x = x.reshape(B, self.num_frames, self.view_num, H, W, C).permute(0, 1, 2, 5, 3, 4).reshape(BTV, C, H, W)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics.float()).to(torch.bfloat16).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature

def prepare_feat_proj_data_lists(
        features, intrinsics, extrinsics, near, far, num_samples
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().detach()
                    @ extrinsics[:, v0].clone().detach().inverse()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
            min_depth
            + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
            * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr




class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            resample: Literal['default', 'up', 'down'] = 'default',
            groups: int = 16,
            eps: float = 1e-5,
            kernel_size: int=3,
            skip_scale: float = 1,  # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2))

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)

        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x


class DownBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int = 1,
            downsample: bool = True,
            attention: bool = True,
            attention_type: list[bool, bool, bool] = [False, False, False],
            attention_heads: int = 16,
            skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(None)
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, intrinsics, c2ws):
        xs = []

        for attn, net in zip(self.attns, self.nets):
            x = net(x)
            if attn:
                x = attn(x, intrinsics, c2ws)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)

        return x, xs


class MidBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_layers: int = 1,
            attention: bool = True,
            attention_heads: int = 16,
            skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(basic_blocks(in_channels, out_dim=in_channels,
                                     geo_flag=True,
                                     layers=1, proposal_w=2,
                                     proposal_h=2, fold_w=4, fold_h=4,
                                     heads=int(in_channels / 32),
                                     head_dim=32))
            if attention:
                attns.append(TCAttention(in_channels, 4,
                                         skip_scale=skip_scale, num_frames=3, view_num=3))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

    def forward(self, x, intri=None, extri=None, depths=None):
        x = self.nets[0](x)
        loss = 0
        for attn, net in zip(self.attns, self.nets[1:]):
            for subnet in net:
                x, depth_loss = subnet(x, intri, extri, depths)
                loss += depth_loss
            if attn:
                x = attn(x)
        return x, loss



class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            prev_out_channels: list,
            out_channels: int,
            num_layers: int = 1,
            upsample: bool = True,
            attention: bool = True,
            attention_type: list[bool, bool, bool] = [False, False, False],
            attention_heads: int = 16,
            skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels[1] if (i == num_layers - 1) else prev_out_channels[0] # (i == num_layers - 1)
            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=1)) #  + cskip
            if attention:
                attns.append(basic_blocks(out_channels, out_dim=out_channels,
                                     geo_flag=False,
                                     layers=1, proposal_w=2,
                                     proposal_h=2, fold_w=4*(2^num_layers), fold_h=4*(2^num_layers),
                                     heads=int(in_channels / 32),
                                     head_dim=32))

            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, xs, intrinsics, c2ws):
        # breakpoint()
        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            # print(x.shape)
            x = net(x)
            if attn:
                for sub_attn in attn:
                    x = sub_attn(x, intrinsics, c2ws)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x

class PositionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 1, # WARN: hardcoded!
        flag: list[bool, bool, bool] = [True, True, True]
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames
        self.view_num = 3
        self.num_samples = 20
        self.near =0.1
        self.far = 100.0
        self.flag = flag

        if flag[0]:
            self.depth_act = lambda x: 0.5 * torch.tanh(x).cuda() + 0.5
            self.position_fusion = nn.Sequential(
                nn.BatchNorm2d(dim + 3),
                nn.Conv2d(dim + 3, dim, kernel_size=9, stride=1, padding=4),
                nn.SiLU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4),
                nn.SiLU()
            )
        if flag[1]:
            self.corr_project = nn.Conv2d(dim + self.num_samples, dim, 3, 1, 1)
            self.corr_fusion = nn.Sequential(
                nn.BatchNorm2d(dim * 2),
                nn.Conv2d(dim * 2, dim, kernel_size=9, stride=1, padding=4),
                nn.SiLU()
            )
        if flag[2]:
            self.view_fusion = basic_blocks(dim, out_dim=dim, layers=2, proposal_w=2,
                                                 proposal_h=2, fold_w=1, fold_h=1,
                                                 heads=int(dim/32),
                                                 head_dim=32)


    def depth2xyz(self, Depth, intrinsics, c2ws, N=3, H=640, W=960, scale=1):

        channel_1 = torch.arange(W).unsqueeze(0).expand(H, -1).repeat(N, 1, 1).to(Depth.device)
        channel_2 = torch.arange(H).unsqueeze(0).expand(W, -1).permute(1, 0).repeat(N, 1, 1).to(Depth.device)
        uv_map = torch.stack((channel_1, channel_2, Depth), dim=-1).to(torch.bfloat16)  # 三张图，H， W， u， v， depth
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
        xyzs = xyzs_world.reshape(N, H, W, 3).permute(0, 3, 1, 2)

        return xyzs

    def forward(self, x, intrinsics, c2ws):
        # x: [B*V, C, H, W]
        BV, C, H, W = x.shape
        B = int(BV / self.view_num)
        if self.flag[0]:
            # posion encoding
            Depth_pre = 255.0 * self.depth_act(x[:, 0, :, :])
            xyz = self.depth2xyz(Depth_pre, intrinsics, c2ws, N=BV, H=H, W=W)
            x = torch.cat([xyz, x], dim=1)

            # feats = feats.reshape(B, self.view_num, C+3, H, W).permute(0, 2, 3, 1, 4).reshape(B, C+3, H, W * self.view_num)
            x = self.position_fusion(x)

        # xyz → u, v
        if self.flag[1]:
            near = torch.tensor(self.near).to(x.device).repeat(B, self.view_num)
            far = torch.tensor(self.far).to(x.device).repeat(B, self.view_num)
            x = x.reshape(B, self.view_num, C, H, W)
            intri = intrinsics.reshape(B, self.view_num, 3, 3)
            exs = c2ws.reshape(B, self.view_num, 4, 4)
            feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = prepare_feat_proj_data_lists(x, intri, exs, near, far, self.num_samples)

            feat01 = feat_comb_lists[0]
            raw_correlation_in_lists = []
            for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                # sample feat01 from feat10 via camera projection
                feat01_warped = warp_with_pose_depth_candidates(
                    feat10,
                    intr_curr,
                    pose_curr,
                    1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                    warp_padding_mode="zeros",
                )  # [B, C, D, H, W]
                # calculate similarity
                raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                    1
                ) / (
                    C**0.5
                )  # [vB, D, H, W]
                raw_correlation_in_lists.append(raw_correlation_in)
            # average all cost volumes
            raw_correlation_in = torch.mean(
                torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
            )  # [vxb d, h, w]
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
            cost_volume = self.corr_project(raw_correlation_in).reshape(self.view_num, B, C, H, W).permute(1, 0, 2, 3, 4)
            x = self.corr_fusion(torch.cat([cost_volume, x], dim=2).reshape(BV, C*2, H, W))


        if self.flag[2]:
            #### view coss
            res = x
            x = x.reshape(B, self.view_num, C, H, W).permute(0, 2, 3, 4, 1).reshape(B, C, H, W * self.view_num)
            B = int(BV/self.view_num)
            W = W * self.view_num
            x = x.reshape(B, C, H, W)
            x = self.view_fusion(x)
            W = int(W / self.view_num)
            x = x.reshape(B, C, H, W, self.view_num).permute(0, 4, 1, 2, 3).reshape(BV, C, H, W)
            if self.residual:
                x = (x + res) * self.skip_scale

        return x


# it could be asymmetric!
class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_channels: Tuple[int, ...] = (64, 128, 256, 256),
            down_attention: Tuple[bool, ...] = (False, False, False, False),
            down_attention_type: Tuple[list, ...] = (
                    [False, False, False], [False, False, False], [False, False, False], [False, False, False]),
            mid_attention: bool = True,
            up_channels: Tuple[int, ...] = (256, 256, 128),
            up_attention: Tuple[bool, ...] = (True, True, False, False),
            up_attention_type: Tuple[list, ...] = (
                    [False, False, True], [False, False, True], [False, False, True], [False, False, False]),
            layers_per_block: int = 1,
            skip_scale: float = np.sqrt(0.5),
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout,
                num_layers=layers_per_block,
                downsample=(i != len(down_channels) - 1),  # not final layer
                attention=down_attention[i],
                attention_type=down_attention_type[i],
                skip_scale=skip_scale,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = [down_channels[max(-1 - i, -len(down_channels))], down_channels[max(-2 - i, -len(down_channels))]]  # for assymetric
            # breakpoint()
            up_blocks.append(UpBlock(
                cin, cskip, cout,
                num_layers=layers_per_block + 1,  # one more layer for up
                upsample=(i != len(up_channels) - 1),  # not final layer
                attention=up_attention[i],
                attention_type=up_attention_type[i],
                skip_scale=skip_scale,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, x, intrinsics, c2ws, depths):
        x = x.to(torch.bfloat16)
        x = self.conv_in(x)

        # down
        xss = [x]
        for block in self.down_blocks:
            x, xs = block(x, intrinsics, c2ws)
            xss.extend(xs)

        # mid
        x, loss= self.mid_block(x, intrinsics, c2ws, depths)
        # breakpoint()
        # up
        for block in self.up_blocks:

            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            x = block(x, xs, intrinsics, c2ws)

        return x, loss
