import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize


class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self,
                 spec_size,
                 feature_strides,
                 in_channels,
                 channels,
                 num_classes,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super(FPNHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.spec_size = spec_size
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.out_layer = ConvModule(
                            self.channels,
                            2,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)

    def forward(self, x):

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = resize(input=output,
                        size=self.spec_size,
                        mode='bilinear',
                        align_corners=self.align_corners)

        return torch.sigmoid(self.out_layer(output))
