import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from einops import rearrange, repeat
from timm.models.layers import DropPath
import math
"""
"""
class SQAttention(nn.Module):
    """
    Spatial-query Attention
    raw:https://github.com/wpy1999/SAT/blob/main/Model/SAT.py

    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., vis=False):
        super().__init__()
        self.vis = vis
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = nn.Sigmoid()(attn[:, :, -1, :].unsqueeze(2).mean(1).unsqueeze(1))

        attn = attn.softmax(dim=-1)
        attn = attn * mask

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FRN(nn.Module):
    """
    Feature Refinement Feed-forward Network
    raw: https://github.com/joshyZhou/AST/blob/main/model.py
    """
    def __init__(self, dim=512, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # x: 256 B 512, B=24

        # bs x hw x c
        hw, bs, c = x.size()
        hh = int(math.sqrt(hw))

        # spatial restore
        x = rearrange(x, ' (h w) b (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w ->  (h w) b c', h=hh, w=hh)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' (h w) b (c) -> b c h w ', h=hh, w=hh)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w ->  (h w) b c', h=hh, w=hh)
        x = x_1 * x_2

        x = self.linear2(x)
        # x = self.eca(x)

        return x


class SAFR_Block(nn.Module):
    """
    spatial-aware feature refinement module

    """
    def __init__(self, dim=512, num_heads=8, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SQAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            vis=vis)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.feaRefine = FRN()
        self.norm2 = norm_layer(dim)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.feaRefine(self.norm2(x)))
        return x

if __name__ == '__main__':
    m=SAFR_Block()
    input=torch.randn(256, 24, 512)
    output=m(input)
    print(output.shape)
