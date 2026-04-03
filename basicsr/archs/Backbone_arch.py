import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
import sys
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import warnings


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SS2D2(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.1,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.conv2d_2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # d = self.act(self.conv2d_2(d))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  ## Local enhancement
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.1,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.conv2d_2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        d = self.act(self.conv2d_2(d))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4 + d.contiguous().view(B, -1, H * W)  ## Local enhancement
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            drop_rate: float = 0.1,
            d_state: int = 16,
            expand: float = 2.,
            img_size: int = 224,
            patch_size: int = 4,
            embed_dim: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)

        self.ss2d = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ffn = FeedForward(hidden_dim, expand, bias=True)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)

        self.conv2d = nn.Conv2d(int(hidden_dim * expand), int(hidden_dim * expand), kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim, bias=False)

    def forward(self, inputs):
        input, illum = inputs
        input_size = (input.shape[2], input.shape[3])
        input = self.patch_embed(input)
        input = self.pos_drop(input)
        illum = F.gelu(self.conv2d(illum))
        B, L, C = input.shape
        input = input.view(B, *input_size, C).contiguous()  # [B,H,W,C]
        x = input + self.drop_path(self.ss2d(self.ln_1(input), illum))
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, input_size) + self.ffn(self.patch_unembed(self.ln_2(x), input_size), illum)
        return (x, illum)


class VSSBlock1(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            drop_rate: float = 0.1,
            d_state: int = 16,
            expand: float = 2.,
            img_size: int = 224,
            patch_size: int = 4,
            embed_dim: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)

        self.ss2d = SS2D2(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ffn = FeedForward2(hidden_dim, expand, bias=True)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)

        self.conv2d = nn.Conv2d(int(hidden_dim * expand), int(hidden_dim * expand), kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim, bias=False)

    def forward(self, inputs):
        input = inputs
        input_size = (input.shape[2], input.shape[3])
        input = self.patch_embed(input)
        input = self.pos_drop(input)
        # illum = F.gelu(self.conv2d(illum))
        B, L, C = input.shape
        input = input.view(B, *input_size, C).contiguous()  # [B,H,W,C]
        x = input + self.drop_path(self.ss2d(self.ln_1(input)))
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, input_size) + self.patch_unembed(self.ln_2(x), input_size)
        return x


class FeedForward(nn.Module):  ## Implicit Retinex-Aware
    def __init__(self, dim, expand, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * expand)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                 groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, 2, kernel_size=3, padding=1, bias=bias)
        self.dwconv4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.act = nn.Sigmoid()

    def forward(self, x_in, illum):
        x = self.project_in(x_in)
        attn1 = self.dwconv(x)
        attn2 = self.dwconv2(attn1)
        illum1, illum2 = self.dwconv3(illum).chunk(2, dim=1)
        attn = attn1 * self.act(illum1) + attn2 * self.act(illum2)
        x = x + attn * x
        x = F.gelu(self.dwconv4(x))
        x = self.project_out(x)
        return x


class FeedForward2(nn.Module):  ## Implicit Retinex-Aware
    def __init__(self, dim, expand, bias):
        super(FeedForward2, self).__init__()
        hidden_features = int(dim * expand)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                 groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, 2, kernel_size=3, padding=1, bias=bias)
        self.dwconv4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.act = nn.Sigmoid()

    def forward(self, x_in):
        x = self.project_in(x_in)
        attn1 = self.dwconv(x)
        attn2 = self.dwconv2(attn1)
        attn = attn1 + attn2
        # illum1,illum2 = self.dwconv3(illum).chunk(2, dim=1)
        # attn = attn1*self.act(illum1)+attn2*self.act(illum2)
        x = x + attn * x
        x = F.gelu(self.dwconv4(x))
        x = self.project_out(x)
        return x


class R(nn.Module):
    def __init__(self, channel=32, layers=6):
        super(R, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)

    def forward(self, illu, input_low_img, refl, mu1, mu2):
        A = illu ** 2 + mu2 * input_low_img + 1e-6
        # A_inv = torch.inverse(A)
        refl_mean = torch.mean(refl, dim=1, keepdim=True)
        input_low_img_mean = torch.mean(input_low_img, dim=1, keepdim=True)
        b = illu * input_low_img + mu2 * refl
        R = b / A
        # b = illu * input_low_img + mu2 * refl + mu1 * self.convs(torch.sign(self.convs(input_low_img_mean)-self.convs(refl_mean)))
        # R = b / A
        return R


class L(nn.Module):
    def __init__(self, channel=32, layers=6):
        super(L, self).__init__()

        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 3, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        self.w = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

    def forward(self, illu, input_low_img, refl, lambda1, lambda2):
        ones_matrix = torch.ones_like(illu)
        illu = torch.mean(illu, dim=1, keepdim=True)
        A = refl ** 2 + lambda2 * ones_matrix + lambda1 * (self.w ** 2) * (self.convs(illu) ** 2) + 1e-6
        # A_inv = torch.inverse(A)
        b = refl * input_low_img + lambda2 * illu
        L = b / A
        # L = torch.mean(L, dim=1, keepdim=True)
        return L


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, ll, lh, hl, hh):
        # 根据公式重构原图分块
        x1 = (ll - lh - hl + hh) / 4.0
        x2 = (ll + lh - hl - hh) / 4.0
        x3 = (ll - lh + hl - hh) / 4.0
        x4 = (ll + lh + hl + hh) / 4.0

        B, C, H, W = ll.shape
        # 重构后的图像尺寸为 [B, C, H*2, W*2]
        out = torch.zeros(B, C, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x1  # 偶数行偶数列
        out[:, :, 1::2, 0::2] = x2  # 奇数行偶数列
        out[:, :, 0::2, 1::2] = x3  # 偶数行奇数列
        out[:, :, 1::2, 1::2] = x4  # 奇数行奇数列

        return out


class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size // 2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
        )

    def forward(self, input):
        return self.basic_unit(input)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class UNet_BilateralFilter_mask(nn.Module):
    def __init__(self, in_channels=4, channels=6, out_channels=1):
        super(UNet_BilateralFilter_mask, self).__init__()
        self.convpre = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.conv1 = UNetConvBlock(channels, channels)
        self.down1 = nn.Conv2d(channels, 2 * channels, stride=2, kernel_size=2)
        self.conv2 = UNetConvBlock(2 * channels, 2 * channels)
        self.down2 = nn.Conv2d(2 * channels, 4 * channels, stride=2, kernel_size=2)
        self.conv3 = UNetConvBlock(4 * channels, 4 * channels)

        self.Global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * channels, 4 * channels, 1, 1, 0))
        self.context_g = UNetConvBlock(8 * channels, 4 * channels)

        self.context2 = UNetConvBlock(2 * channels, 2 * channels)
        self.context1 = UNetConvBlock(channels, channels)

        self.merge2 = nn.Sequential(nn.Conv2d(6 * channels, 4 * channels, 1, 1, 0),
                                    CALayer(4 * channels, 4),
                                    nn.Conv2d(4 * channels, 2 * channels, 3, 1, 1)
                                    )
        self.merge1 = nn.Sequential(nn.Conv2d(3 * channels, channels, 1, 1, 0),
                                    CALayer(channels, 2),
                                    nn.Conv2d(channels, channels, 3, 1, 1)
                                    )

        self.conv_last = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(self.convpre(x))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))

        x_global = self.Global(x3)
        _, _, h, w = x3.size()
        x_global = x_global.repeat(1, 1, h, w)
        x3 = self.context_g(torch.cat([x_global, x3], 1))

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x2 = self.context2(self.merge2(torch.cat([x2, x3], 1)))

        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x1 = self.context1(self.merge1(torch.cat([x1, x2], 1)))

        xout = self.conv_last(x1)

        return xout, x3


class UNetConvBlock_fre(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock_fre, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock_fre(self.split_len2, self.split_len1)
        self.G = UNetConvBlock_fre(self.split_len1, self.split_len2)
        self.H = UNetConvBlock_fre(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc, nc // 2)

    def forward(self, x):
        return x + self.block(x)


class FreBlockSpa(nn.Module):
    def __init__(self, nc):
        super(FreBlockSpa, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1, groups=nc))

    def forward(self, x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class FreBlockCha(nn.Module):
    def __init__(self, nc):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=1, padding=0, stride=1))

    def forward(self, x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out


class SpatialFuse(nn.Module):
    def __init__(self, in_nc):
        super(SpatialFuse, self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockSpa(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc, in_nc, 3, 1, 1)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 3, 1, 1)

    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x, x_freq_spatial], 1)
        x_out = self.cat(xcat)

        return x_out + xori


class ChannelFuse(nn.Module):
    def __init__(self, in_nc):
        super(ChannelFuse, self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockCha(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = self.frequency_spatial(x_freq_spatial)
        xcat = torch.cat([x, x_freq_spatial], 1)
        x_out = self.cat(xcat)

        return x_out + xori


class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.spa = SpatialFuse(nc)
        self.cha = ChannelFuse(nc)

    def forward(self, x):
        x = self.spa(x)
        x = self.cha(x)

        return x


class ProcessNet(nn.Module):
    def __init__(self, nc):
        super(ProcessNet, self).__init__()
        self.conv0 = nn.Conv2d(nc, nc, 3, 1, 1)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc, nc * 2, stride=2, kernel_size=2, padding=0)
        self.conv2 = ProcessBlock(nc * 2)
        self.downsample2 = nn.Conv2d(nc * 2, nc * 3, stride=2, kernel_size=2, padding=0)
        self.conv3 = ProcessBlock(nc * 3)
        self.up1 = nn.ConvTranspose2d(nc * 5, nc * 2, 1, 1)
        self.conv4 = ProcessBlock(nc * 2)
        self.up2 = nn.ConvTranspose2d(nc * 3, nc * 1, 1, 1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc, nc, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3, size=(x12.size()[2], x12.size()[3]), mode='bilinear'), x12], 1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4, size=(x01.size()[2], x01.size()[3]), mode='bilinear'), x01], 1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)

        return xout


class InteractNet(nn.Module):
    def __init__(self, inchannel, nc, outchannel):
        super(InteractNet, self).__init__()
        self.extract = nn.Conv2d(inchannel, nc, 1, 1, 0)
        self.process = ProcessNet(nc)
        self.recons = nn.Conv2d(nc, outchannel, 1, 1, 0)

    def forward(self, x):
        x_f = self.extract(x)
        x_f = self.process(x_f) + x_f
        y = self.recons(x_f)

        return y


class Illumination_Estimator(nn.Module):


    def __init__(self, n_fea_middle, img_size, n_fea_in=4, n_fea_out=3):

        super(Illumination_Estimator, self).__init__()

        self.VSSB_L = nn.Sequential(*[VSSBlock1(
            hidden_dim=96, norm_layer=nn.LayerNorm, d_state=32, expand=2, img_size=img_size,
            patch_size=1, embed_dim=96)])

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)


        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)


        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)


    def forward(self, img, illu_hat):

        mean_c = illu_hat.mean(dim=1).unsqueeze(1)

        input = torch.cat([img, mean_c], dim=1)  # 对应

        x_1 = self.conv1(input)

        illu_fea = self.VSSB_L(x_1)

        illu_map = self.conv2(illu_fea)

        return illu_fea, illu_map


@ARCH_REGISTRY.register()
class Backbone(nn.Module):
    def __init__(self, nf=64,
                 img_size=128,
                 patch_size=1,
                 embed_dim=64,
                 depths=(1, 2, 2, 2, 2, 2),
                 d_state=64,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 num_layer=3):
        super(Backbone, self).__init__()

        self.nf = nf
        self.depths = depths

        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=False)
        self.conv_first_1_fea = nn.Conv2d(5, int(nf * mlp_ratio), 3, 1, 1)

        self.VSSB_1 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf, norm_layer=norm_layer, d_state=d_state, expand=mlp_ratio, img_size=img_size,
            patch_size=patch_size, embed_dim=embed_dim) for i in range(self.depths[0])])

        self.conv_first_2 = nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)
        self.conv_first_2_fea = nn.Conv2d(5, int(nf * 2 * mlp_ratio), 3, 1, 1)
        self.VSSB_2 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf * 2, norm_layer=norm_layer, d_state=d_state, expand=mlp_ratio, img_size=img_size // 2,
            patch_size=patch_size, embed_dim=embed_dim * 2) for i in range(self.depths[1])])

        self.conv_first_3 = nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)
        self.conv_first_3_fea = nn.Conv2d(5, int(nf * 4 * mlp_ratio), 3, 1, 1)
        self.VSSB_3 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf * 4, norm_layer=norm_layer, d_state=d_state, expand=mlp_ratio, img_size=img_size // 4,
            patch_size=patch_size, embed_dim=embed_dim * 4) for i in range(self.depths[2])])

        self.conv_first_4 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False)
        self.conv_first_4_fea = nn.Conv2d(5, int(nf * 4 * mlp_ratio), 3, 1, 1)
        self.VSSB_4 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf * 4, norm_layer=norm_layer, d_state=d_state, expand=mlp_ratio, img_size=img_size // 4,
            patch_size=patch_size, embed_dim=embed_dim * 4) for i in range(self.depths[3])])

        self.upconv1 = nn.ConvTranspose2d(nf * 4, nf * 4 // 2, stride=2,
                                          kernel_size=2, padding=0, output_padding=0)
        self.conv_first_5 = nn.Conv2d(nf * 4, nf * 4 // 2, 3, 1, 1, bias=False)
        self.conv_first_5_fea = nn.Conv2d(5, int(nf * 2 * mlp_ratio), 3, 1, 1)
        self.VSSB_5 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf * 2, norm_layer=norm_layer, d_state=d_state, expand=mlp_ratio, img_size=img_size // 2,
            patch_size=patch_size, embed_dim=embed_dim * 2) for i in range(self.depths[4])])

        self.upconv2 = nn.ConvTranspose2d(nf * 2, nf * 2 // 2, stride=2,
                                          kernel_size=2, padding=0, output_padding=0)
        self.conv_first_6 = nn.Conv2d(nf * 2, nf * 2 // 2, 3, 1, 1, bias=False)
        self.conv_first_6_fea = nn.Conv2d(5, int(nf * mlp_ratio), 3, 1, 1)
        self.VSSB_6 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf, norm_layer=norm_layer, d_state=d_state, expand=mlp_ratio, img_size=img_size,
            patch_size=patch_size, embed_dim=embed_dim) for i in range(self.depths[5])])

        self.out_embed = nn.Conv2d(nf, 3, 3, 1, 1)


    def forward(self, x_in, illu):
        x_max = torch.max(illu, dim=1, keepdim=True)[0]
        x_mean = torch.mean(illu, dim=1, keepdim=True)
        x_in_cat = torch.cat((x_in, x_max, x_mean), dim=1)
        x_2 = F.avg_pool2d(x_in_cat, kernel_size=2, stride=2)
        x_4 = F.avg_pool2d(x_in_cat, kernel_size=4, stride=4)

        x_conv_1 = self.conv_first_1(x_in)
        illum_conv_1 = self.conv_first_1_fea(x_in_cat)
        vssb_fea_1 = self.VSSB_1((x_conv_1, illum_conv_1))[0]

        x_conv_2 = self.conv_first_2(vssb_fea_1)
        illum_conv_2 = self.conv_first_2_fea(x_2)
        vssb_fea_2 = self.VSSB_2((x_conv_2, illum_conv_2))[0]

        x_conv_3 = self.conv_first_3(vssb_fea_2)
        illum_conv_3 = self.conv_first_3_fea(x_4)
        vssb_fea_3 = self.VSSB_3((x_conv_3, illum_conv_3))[0]

        x_conv_4 = self.conv_first_4(vssb_fea_3)
        illum_conv_4 = self.conv_first_4_fea(x_4)
        vssb_fea_4 = self.VSSB_4((x_conv_4, illum_conv_4))[0]

        up_feat_1 = self.upconv1(vssb_fea_4)
        h = min(up_feat_1.shape[2], vssb_fea_2.shape[2])
        w = min(up_feat_1.shape[3], vssb_fea_2.shape[3])

        up_feat_1 = up_feat_1[:, :, :h, :w]
        vssb_fea_2 = vssb_fea_2[:, :, :h, :w]
        x_cat_1 = torch.cat([up_feat_1, vssb_fea_2], dim=1)

        vssb_fea_5 = self.conv_first_5(x_cat_1)
        illum_conv_5 = self.conv_first_5_fea(x_2)
        if illum_conv_5.shape[-2:] != vssb_fea_5.shape[-2:]:
            illum_conv_5 = F.interpolate(
                illum_conv_5, size=vssb_fea_5.shape[-2:],
                mode="bilinear", align_corners=False
            )
        vssb_fea_5 = self.VSSB_5((vssb_fea_5, illum_conv_5))[0]
        up_feat_2 = self.upconv2(vssb_fea_5)
        h = min(up_feat_2.shape[2], vssb_fea_1.shape[2])
        w = min(up_feat_2.shape[3], vssb_fea_1.shape[3])

        up_feat_2 = up_feat_2[:, :, :h, :w]
        vssb_fea_1 = vssb_fea_1[:, :, :h, :w]
        x_cat_2 = torch.cat([up_feat_2, vssb_fea_1], dim=1)
        vssb_fea_6 = self.conv_first_6(x_cat_2)
        illum_conv_6 = self.conv_first_6_fea(x_in_cat)

        if illum_conv_6.shape[-2:] != vssb_fea_6.shape[-2:]:
            illum_conv_6 = F.interpolate(
                illum_conv_6, size=vssb_fea_6.shape[-2:],
                mode="bilinear", align_corners=False
            )

        vssb_fea_6 = self.VSSB_6((vssb_fea_6, illum_conv_6))[0]
        out = self.out_embed(vssb_fea_6)
        if out.shape[-2:] != x_in.shape[-2:]:
            out = F.interpolate(out, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
        out = out + x_in
        return out







