import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

from src.model.module.conv2d import ConvBlock
from src.model.module.pvt_v2 import Block


class DeOverlapPatchEmbed(nn.Module):
    """ Patch Decoding
    """

    def __init__(self, spec_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        spec_size = to_2tuple(spec_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        assert max(patch_size) > max(stride), "Set larger patch_size than stride"

        self.spec_size = spec_size
        self.patch_size = patch_size
        self.H, self.W = spec_size[0] * stride[0], spec_size[1] * stride[1]
        self.num_patches = self.H * self.W
        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2),
                                       output_padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class DePyramidVisionTransformerV2(nn.Module):
    def __init__(self, spec_size=224, patch_size=[1, (2, 1), 2, (8, 2)], in_chans=3, num_classes=1000,
                 embed_dims=[512, 256, 128, 64],
                 num_heads=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 6, 4, 3], sr_ratios=[1, 2, 4, 8], num_stages=4, linear=False, conv_cfg=None):
        super().__init__()
        spec_size = torch.tensor(spec_size)
        patch_size_t = torch.cumprod(torch.tensor(patch_size), dim=0)

        # conv
        self.conv = ConvBlock(**conv_cfg)

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.transition_patch_embed = DeOverlapPatchEmbed(spec_size=spec_size,
                                                          patch_size=patch_size[0],
                                                          stride=[(patch_size[0][0]+1)//2, (patch_size[0][1]+1)//2],
                                                          in_chans=embed_dims[0],
                                                          embed_dim=embed_dims[1]
                                                          )

        for i in range(1, num_stages):
            block = nn.ModuleList([Block(
                dim=embed_dims[i] * 2, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i]*2)
            patch_embed = DeOverlapPatchEmbed(spec_size=spec_size * patch_size_t[i - 1],
                                              patch_size=patch_size[i],
                                              stride=[(patch_size[i][0]+1)//2, (patch_size[i][1]+1)//2],
                                              in_chans=2*embed_dims[i],
                                              embed_dim=conv_cfg["in_channels"] // 2 if i == len(embed_dims)-1 else
                                              embed_dims[i + 1]
                                              )
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, encoder_output):
        x = encoder_output[0]
        B = x.shape[0]

        x, H, W = self.transition_patch_embed(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        for i in range(1, self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = torch.cat((x, encoder_output[i]), dim=1)
            x = x.flatten(2).transpose(1, 2)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            x, H, W = patch_embed(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = torch.cat((x, encoder_output[-1]), dim=1)
        x = self.conv(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size)).contiguous()
        out_dict[k] = v

    return out_dict


@register_model
def pvt_v2_b0(pretrained=False, **kwargs):
    model = DePyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model

#
# @register_model
# def pvt_v2_b1(pretrained=False, **kwargs):
#     model = DePyramidVisionTransformerV2(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     return model
#
#
# @register_model
# def pvt_v2_b2(pretrained=False, **kwargs):
#     model = PyramidVisionTransformerV2(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
#     model.default_cfg = _cfg()
#
#     return model
#
#
# @register_model
# def pvt_v2_b3(pretrained=False, **kwargs):
#     model = PyramidVisionTransformerV2(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     return model
#
#
# @register_model
# def pvt_v2_b4(pretrained=False, **kwargs):
#     model = PyramidVisionTransformerV2(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     return model
#
#
# @register_model
# def pvt_v2_b5(pretrained=False, **kwargs):
#     model = PyramidVisionTransformerV2(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#
#     return model
#
#
# @register_model
# def pvt_v2_b2_li(pretrained=False, **kwargs):
#     model = PyramidVisionTransformerV2(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
#     model.default_cfg = _cfg()
#
#     return model
