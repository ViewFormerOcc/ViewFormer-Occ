import torch.nn as nn
import torch
import torch.nn.functional as F

from mmcv.runner.base_module import BaseModule

from mmdet.models import BACKBONES

from mmdet3d.models import builder

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() == 4:
            return x.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            return x.permute(0, 4, 1, 2, 3)

class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() == 4:
            return x.permute(0, 2, 3, 1)
        elif x.dim() == 5:
            return x.permute(0, 2, 3, 4, 1)

def custom_build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def custom_build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

# We use LN as norm layer and GELU as act layer (original: BN & Relu)
class Custom_BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 need_lateral_feat=False,
                 drop_prob=0.,
                 dcn=None,
                 plugins=None,
                 act_cfg='GELU',
                 init_cfg=None):
        super(Custom_BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        from mmcv.cnn import build_conv_layer

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.norm1 = custom_build_norm_layer(planes, 'LN', in_format='channels_first', out_format='channels_first')
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.norm2 = custom_build_norm_layer(planes, 'LN', in_format='channels_first', out_format='channels_first')

        #self.relu = nn.ReLU(inplace=True)
        self.act = custom_build_act_layer(act_cfg)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

        self.need_lateral_feat = need_lateral_feat
        self.drop_connection = DropBlock2D(drop_prob=drop_prob, block_size=3) if drop_prob > 0. else nn.Identity()

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            return self.drop_connection(out) + identity

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        if self.need_lateral_feat:
            return self.act(out), out

        out = self.act(out)

        return out


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_dim, out_dim, pad=0):
        super().__init__()
        self.down_layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, padding=pad, bias=False),
            custom_build_norm_layer(out_dim, 'LN', in_format='channels_first', out_format='channels_first')
        )

    def forward(self, x):
        return self.down_layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_cfg='GELU', out_pad=0, need_lateral_feat=False):
        super(UpBlock, self).__init__()

        in_dim = int(in_dim)
        out_dim = int(out_dim)

        self.need_lateral_feat = need_lateral_feat

        self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(
                                    in_dim,
                                    out_dim,
                                    kernel_size=(2, 2),
                                    stride=(2, 2),
                                    padding=out_pad,
                                    output_padding=out_pad,
                                    bias=False),
                    custom_build_norm_layer(out_dim, 'LN', in_format='channels_first', out_format='channels_first'),
                )

        self.nonlinearity = custom_build_act_layer(act_cfg)

        self.feat_layer = Custom_BasicBlock(out_dim, out_dim, need_lateral_feat=need_lateral_feat)

    def forward(self, x, lateral):
        x = self.nonlinearity(self.upsample(x) + lateral)
        if self.need_lateral_feat:
            _, x = self.feat_layer(x)
        else:
            x = self.feat_layer(x)
        return x


class BasicLayer(nn.Module):
    """ A basic layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, down_ex_scale=2,
                 drop_prob=[0.],
                 need_lateral_feat=False,
                 downsample=None,
                 down_pad=0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.need_lateral_feat = need_lateral_feat

        blocks = []
        for i in range(depth):
            need_lateral = need_lateral_feat if i ==(depth - 1) else False
            blocks.append(
                Custom_BasicBlock(
                    inplanes=dim,
                    planes=dim,
                    need_lateral_feat=need_lateral,
                    drop_prob=drop_prob[i],
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(in_dim=dim, out_dim=int(down_ex_scale*dim), pad=down_pad)
        else:
            self.downsample = None

    def forward(self, x):
        if self.need_lateral_feat:
            x, x_lateral = self.blocks(x)
        else:
            x = self.blocks(x)
            x_lateral = None

        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_lateral


@BACKBONES.register_module()
class UNET_CNN(nn.Module):

    def __init__(self,
                 in_channels,
                 num_blocks=[2, 2, 3],
                 down_ex_scales=[2, 2, 2],
                 act_cfg='GELU',
                 drop_cfg=dict(type='dropblock', block_size=3, drop_prob=0.1),
                 down_pads=[0, 0, 0],
                 ):
        super(UNET_CNN, self).__init__()

        self.act_cfg = act_cfg

        up_pads = down_pads[::-1]

        layer_channels = []
        layer_channels.append(in_channels)

        for scale in down_ex_scales:
            next_channels = int(layer_channels[-1] * scale)
            layer_channels.append(next_channels)


        dpr = [x.item() for x in torch.linspace(0, drop_cfg['drop_prob'], sum(num_blocks)+2)]

        self.in_planes = in_channels
        self.stage1 = self._make_stage(num_blocks[0], down_ex_scale=down_ex_scales[0], need_lateral_feat=True, drop_prob=dpr[sum(num_blocks[:0]):sum(num_blocks[:0 + 1])], downsample=PatchMerging, down_pad=down_pads[0])
        self.stage2 = self._make_stage(num_blocks[1], down_ex_scale=down_ex_scales[1], need_lateral_feat=True, drop_prob=dpr[sum(num_blocks[:1]):sum(num_blocks[:1 + 1])], downsample=PatchMerging, down_pad=down_pads[1])

        if len(num_blocks) == 3:
            self.stage3 = self._make_stage(num_blocks[2], down_ex_scale=down_ex_scales[2], need_lateral_feat=True, drop_prob=dpr[sum(num_blocks[:2]):sum(num_blocks[:2 + 1])], downsample=PatchMerging, down_pad=down_pads[2])

        self.extra_conv = nn.Sequential(
            Custom_BasicBlock(inplanes=layer_channels[-1], planes=layer_channels[-1], drop_prob=dpr[-2]),
            Custom_BasicBlock(inplanes=layer_channels[-1], planes=layer_channels[-1], drop_prob=dpr[-1]),
        )

        layer_start_idx = -1
        pad_start_idx = 0
        if len(num_blocks) == 3:
            self.decoder0 = UpBlock(layer_channels[layer_start_idx  ], layer_channels[layer_start_idx-1], act_cfg=act_cfg, out_pad=up_pads[pad_start_idx  ])
            layer_start_idx = layer_start_idx - 1
            pad_start_idx = pad_start_idx + 1

        self.decoder1 = UpBlock(layer_channels[layer_start_idx  ], layer_channels[layer_start_idx-1], act_cfg=act_cfg, out_pad=up_pads[pad_start_idx  ])
        self.decoder2 = UpBlock(layer_channels[layer_start_idx-1], layer_channels[layer_start_idx-2], act_cfg=act_cfg, out_pad=up_pads[pad_start_idx+1],need_lateral_feat=True)

    def init_weights(self, pretrained=None):
        return

    def _make_stage(self, num_blocks, down_ex_scale=2, need_lateral_feat=False, drop_prob=[0.], downsample=None, down_pad=None):
        stage_layer = BasicLayer(dim=self.in_planes, depth=num_blocks, down_ex_scale=down_ex_scale,
                                 drop_prob=drop_prob,
                                 need_lateral_feat=need_lateral_feat,
                                 downsample=downsample,
                                 down_pad=down_pad)
        self.in_planes = int(self.in_planes * down_ex_scale)
        return stage_layer

    def forward(self, x):

        x, x1_lateral = self.stage1(x)
        x, x2_lateral = self.stage2(x)
        if hasattr(self, 'stage3'):
            x, x3_lateral = self.stage3(x)
        x = self.extra_conv(x)

        if hasattr(self, 'decoder0'):
            x = self.decoder0(x, x3_lateral)
        x = self.decoder1(x, x2_lateral)
        x = self.decoder2(x, x1_lateral)

        return x


class UNET_DecoderOnly(nn.Module):

    def __init__(self,
                 num_levels=3,
                 channles=[256, 256, 256],
                 num_classes=18,
                 loss_weight=0.4,
                 ):
        super(UNET_DecoderOnly, self).__init__()

        from mmdet.models.builder import build_loss

        self.num_classes = num_classes
        self.loss_cls = build_loss(dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=loss_weight))
        self.loss_lovasz = build_loss(dict(
                     type='LovaszLoss',
                     loss_weight=loss_weight))

        start_idx = -1
        if num_levels == 4:
            self.decoder0 = UpBlock(channles[start_idx], channles[start_idx-1])
            start_idx -= 1
        self.decoder1 = UpBlock(channles[start_idx], channles[start_idx-1])
        self.decoder2 = UpBlock(channles[start_idx-1], channles[start_idx-2], need_lateral_feat=True)

        self.predictor = nn.Conv2d(channles[start_idx-2], num_classes, 1)

    def forward(self, feat_list, img_metas, return_loss=True):

        if feat_list[0].dim() == 5:
            tmp_feat_list = []
            for i in range(len(feat_list)):
                tmp_feat_list.append(feat_list[i].flatten(0, 1))
            feat_list = tmp_feat_list

        x = feat_list[-1]
        lateral = feat_list[:-1]

        if hasattr(self, 'decoder0'):
            x = self.decoder0(x, lateral[-1])
            del lateral[-1]
        x = self.decoder1(x, lateral[-1])
        x = self.decoder2(x, lateral[-2])

        semantic = self.predictor(x)

        if return_loss:
            semantic_label = torch.stack([torch.stack(each['pixel_wise_label'], dim=0) for each in img_metas], dim=0)[..., -1].flatten(0, 1)
            semantic = F.interpolate(semantic, size=semantic_label.shape[-2:], mode='bilinear', align_corners=True)

            semantic = semantic.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
            semantic_label = semantic_label.view(-1).type_as(semantic).long()

            loss_cls = self.loss_cls(semantic, semantic_label)
            loss_lovasz = self.loss_lovasz(semantic, semantic_label)
            loss_cls = loss_cls + loss_lovasz
            return loss_cls
        else:
            return semantic



class DropBlock2D(nn.Module):
    """ DropBlock: a regularization method for convolutional neural networks.

        DropBlock is a form of structured dropout, where units in a contiguous
        region of a feature map are dropped together. DropBlock works better than
        dropout on convolutional layers due to the fact that activation units in
        convolutional layers are spatially correlated.
        See https://arxiv.org/pdf/1810.12890.pdf for details.
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            height = x.size(2)
            width = x.size(3)

            # get gamma value
            gamma = self._compute_gamma(x)

            # Forces the block to be inside the feature map.
            valid_block_center = torch.zeros(height, width, device=x.device).float()
            valid_block_center[int(self.block_size // 2):(height - (self.block_size - 1) // 2), int(self.block_size // 2):(width - (self.block_size - 1) // 2)] = 1.0
            valid_block_center = valid_block_center.unsqueeze(0).unsqueeze(0)

            # sample mask
            mask = (torch.rand(x.shape, dtype=x.dtype, device=x.device) < gamma).float()
            mask *= valid_block_center

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            keep_mask = 1 - block_mask
            percent_ones = keep_mask.sum() / float(keep_mask.numel())

            out = x * keep_mask
            out = out / percent_ones

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        return block_mask

    def _compute_gamma(self, x):
        height, width = x.size(2), x.size(3)
        total_size = width * height
        return self.drop_prob / (self.block_size ** 2) * total_size / (( width -self.block_size + 1)*( height -self.block_size + 1))