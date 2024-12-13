import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim, l2_loss, compute_depth


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            resample: Literal['default', 'up', 'down'] = 'default',
            groups: int = 16,
            eps: float = 1e-5,
            kernel_size: int=5,
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


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.dim = dim
        self.norm = GroupNorm(heads * head_dim)
        self.act = F.silu
        if dim != heads * head_dim:
            self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=7, stride=1, padding=3)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center

    def forward(self, x, xyz):  # [b,c,w,h]
        value = self.v(x)
        if self.dim != self.heads * self.head_dim:
            x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
            if xyz != None:
                xyz = rearrange(xyz, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)

        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        if xyz != None:
            xyz_center = self.centers_proposal(xyz)
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        if xyz==None:
            sim = torch.sigmoid(
                self.sim_beta +
                self.sim_alpha * pairwise_cos_sim(
                    centers.reshape(b, c, -1).permute(0, 2, 1),
                    x.reshape(b, c, -1).permute(0, 2, 1)
                )
            )  # [B,M,N]
        else:
            xyz = rearrange(xyz, 'b c w h -> b c (w h)')
            xyz_center = rearrange(xyz_center, 'b c w h -> b c (w h)')
            sim = torch.sigmoid(self.sim_beta +
                self.sim_alpha * pairwise_cos_sim(
                    centers.reshape(b, c, -1).permute(0, 2, 1),
                    x.reshape(b, c, -1).permute(0, 2, 1)
                )
            )
            sim_geo = torch.sigmoid(self.sim_beta +
                                self.sim_alpha * pairwise_cos_sim(
                xyz_center.permute(0, 2, 1),
                xyz.permute(0, 2, 1)
            ))
            sim_geo = sim_geo.repeat(int(sim.shape[0]/sim_geo.shape[0]), 1, 1)
            sim = torch.mul(sim, sim_geo**2)


        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        dense_mask = torch.ones_like(sim)  # binary #[B,M,N]
        dense_mask.scatter_(1, sim_max_idx, 1.)
        split_mask = 1 - dense_mask
        denss_sim = sim * dense_mask
        split_sim = sim * split_mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        x2 = rearrange(x, 'b c w h -> b (w h) c')
        # aggregate step, out shape [B,M,D]

        split_out = ((value2.unsqueeze(dim=1) * split_sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                split_sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]
        centers = rearrange(centers, 'b c w h -> b (w h) c')
        denss_out = ((x2.unsqueeze(dim=1) * denss_sim.unsqueeze(dim=-1)).sum(dim=2) + centers) / (
                denss_sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]
        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            # dispatch step, return to each point in a cluster
            denss_out = (denss_out.unsqueeze(dim=2) * denss_sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            denss_out = rearrange(denss_out, "b (w h) c -> b c w h", w=w)
            split_out = (split_out.unsqueeze(dim=2) * split_sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            split_out = rearrange(split_out, "b (w h) c -> b c w h", w=w)
            out = denss_out + split_out
        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.act(self.proj(self.norm(out)))
        return out


class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


class PD_Block(nn.Module):

    def __init__(self, dim, out_dim, mlp_ratio=1.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 geo_flag=False,
                 # for context-cluster
                 proposal_w=2, proposal_h=2,
                 fold_w=2, fold_h=2,
                 heads=4, head_dim=24,
                 view_num=3,
                 time_length=3,
                 return_center=False):
        super().__init__()

        self.view_num = view_num
        self.time_length = time_length
        self.out_dim = out_dim
        self.dim = dim

        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=out_dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim, return_center=False)
        self.ResnetBlock = ResnetBlock(out_dim, out_dim)
        self.geo_flag = geo_flag
        self.con_alpha = nn.Parameter(torch.zeros(1))
        if dim !=out_dim:
            self.reduce = nn.Sequential(nn.BatchNorm2d(dim),
                                        nn.Conv2d(dim, out_dim, groups=int(min(dim, out_dim)/4), kernel_size=1),
                                        nn.SiLU())
        if self.geo_flag == True:
            self.depth_act = lambda x: 0.5 * torch.tanh(x).cuda() + 0.5
            self.position_fusion = nn.Sequential(
                nn.BatchNorm2d(dim + 3),
                nn.Conv2d(dim + 3, dim, kernel_size=9, stride=1, padding=4),
                nn.SiLU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4),
                nn.SiLU()
            )

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

    def forward(self, x, intri=None, extri=None, depths=None):
        BTV, C, H, W = x.shape
        B = int(BTV / self.view_num)
        if self.geo_flag:
            Depth_pre = 255.0 * self.depth_act(x[:, 0, :, :])
            H_d, W_d = depths.shape[-2:]
            depths = depths.reshape(BTV, 1, H_d, W_d)
            depths = depths.to(torch.bfloat16)
            depths = F.interpolate(depths, size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
            depths = depths.to(torch.bfloat16)
            mask = depths > 0.1
            depths = depths * 255.0
            depths_loss_reg = compute_depth("l2", Depth_pre[mask], depths[mask]) / 255.0
            xyz = self.depth2xyz(Depth_pre, intri, extri, N=BTV, H=H, W=W)
            geo = torch.cat([xyz/255.0, x], dim=1)
            # feats = feats.reshape(B, self.view_num, C+3, H, W).permute(0, 2, 3, 1, 4).reshape(B, C+3, H, W * self.view_num)
            x = x + self.position_fusion(geo)
            xyz = xyz.reshape(B, self.view_num, 3, H, W).permute(0, 2, 3, 4, 1).reshape(B, 3, H, W * self.view_num)
        else:
            xyz = None
        x = x.reshape(B, self.view_num, C, H, W).permute(0, 2, 3, 4, 1).reshape(B, C, H, W * self.view_num)
        if self.dim !=self.out_dim:
            x = self.reduce(x) + self.con_alpha * self.token_mixer(self.norm1(x))
        else:
            x = x + self.con_alpha * self.token_mixer(self.norm1(x), xyz)
        x = self.ResnetBlock(x)
        x = x.reshape(B, self.out_dim, H, W, self.view_num).permute(0, 4, 1, 2, 3).reshape(BTV, self.out_dim, H, W)
        if self.geo_flag:
            return x, depths_loss_reg
        else:
            return x


if __name__ == '__main__':
    input = torch.rand(2, 64, 320, 300)
    model = PD_Block(
            64, mlp_ratio=16,
            proposal_w=2, proposal_h=2,
            fold_w=2, fold_h=2, heads=4,
            head_dim=24, return_center=False
        )
    out = model(input)
    print(out.size())
