import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from projects.mmdet3d_plugin.models.backbones.resnet import Custom_BasicBlock

from torch.cuda.amp.autocast_mode import autocast
from mmcv.runner import force_fp32
from mmdet.models.builder import build_loss

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

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 norm_cfg):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        #self.bn = BatchNorm(planes)
        self.bn = custom_build_norm_layer(planes, 'LN', in_format='channels_first', out_format='channels_first')
        self.act = nn.GELU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.act(x)

    def _init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, norm_cfg=None):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 norm_cfg=norm_cfg)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 norm_cfg=norm_cfg)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 norm_cfg=norm_cfg)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 norm_cfg=norm_cfg)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            #BatchNorm(mid_channels),
            custom_build_norm_layer(mid_channels, 'LN', in_format='channels_first', out_format='channels_first'),
            nn.GELU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        #self.bn1 = BatchNorm(mid_channels)
        self.bn1 = custom_build_norm_layer(mid_channels, 'LN', in_format='channels_first', out_format='channels_first')
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        return self.dropout(x)

    def _init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.GELU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class Custom_DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels=80,
                 depth_channels=112, with_semantics=False, num_classes=17, d_bound=None,
                 depth_loss_weight=1.0,
                 with_lovasz_loss=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_lovasz=dict(
                     type='LovaszLoss',
                     loss_weight=1.0)):
        super(Custom_DepthNet, self).__init__()

        self.d_bound = d_bound
        self.depth_channels = depth_channels
        self.depth_loss_weight = depth_loss_weight

        self.loss_cls = build_loss(loss_cls)
        self.with_lovasz_loss = with_lovasz_loss
        self.loss_lovasz = build_loss(loss_lovasz)

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            #nn.BatchNorm2d(mid_channels),
            custom_build_norm_layer(mid_channels, 'LN', in_format='channels_first', out_format='channels_first'),
            nn.GELU(),
        )
        #self.context_conv = nn.Conv2d(mid_channels,
        #                              context_channels,
        #                              kernel_size=1,
        #                              stride=1,
        #                              padding=0)
        self.bn = nn.BatchNorm2d(16)
        #self.norm = custom_build_norm_layer(16, 'LN', in_format='channels_first', out_format='channels_first')
        self.depth_mlp = Mlp(16, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        #self.context_mlp = Mlp(28, mid_channels, mid_channels)
        #self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            Custom_BasicBlock(mid_channels, mid_channels),
            Custom_BasicBlock(mid_channels, mid_channels),
            Custom_BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.with_semantics = with_semantics
        if with_semantics:
            self.sem_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                        mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                custom_build_norm_layer(mid_channels, 'LN', in_format='channels_first', out_format='channels_first'),
                nn.GELU(),
                nn.Conv2d(mid_channels,
                          num_classes,
                          kernel_size=1,
                          stride=1,
                          padding=0),
            )

    def forward(self, x, mats_dict, num_cams=6):

        # trick for temporal training
        batch_size = x.size(0)
        x = x[:, :num_cams, ...].view(-1, x.size(2), x.size(3), x.size(4))

        intrins = []
        sensor2ego = []
        for mats in mats_dict:
            intrins.append(torch.from_numpy(np.asarray(mats['cam_intrinsic'])[:num_cams, :3, :3]))
            sensor2ego.append(torch.from_numpy(np.asarray(mats['cam2ego'])[:num_cams, :3, :]))

        intrins = torch.stack(intrins, dim=0)
        sensor2ego = torch.stack(sensor2ego, dim=0)

        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[..., 0, 0],
                        intrins[..., 1, 1],
                        intrins[..., 0, 2],
                        intrins[..., 1, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.flatten(2, 3),
            ],
            -1,
        )

        mlp_input = mlp_input.type_as(x).flatten(0, 1)

        #mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1], 1, 1)).squeeze(-1).squeeze(-1)
        x = self.reduce_conv(x)
        #context_se = self.context_mlp(mlp_input)[..., None, None]
        #context = self.context_se(x, context_se)
        #context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        #return torch.cat([depth, context], dim=1)

        if self.with_semantics:
            sem_preds = self.sem_conv(x)
            return depth.view(batch_size, num_cams, *depth.shape[1:]), sem_preds.view(batch_size, num_cams, *sem_preds.shape[1:])
        else:
            return depth.view(batch_size, num_cams, *depth.shape[1:]), None

    def get_downsampled_view_label(self, view_label, downsample_factor=None):
        if view_label.dim() == 5:
            gt_depths = view_label[..., 0]
            other_labels = view_label[..., 1:]

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B*N, 1, H, W)

        gt_depths = gt_depths.view(B * N, H // downsample_factor,
                                   downsample_factor, W // downsample_factor,
                                   downsample_factor, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample_factor * downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths, nearst_idx = torch.min(gt_depths_tmp, dim=-1)

        gt_depths = gt_depths.view(B * N, H // downsample_factor,
                                   W // downsample_factor)

        gt_depths = (gt_depths -
                    (self.d_bound[0] - self.d_bound[2])) / self.d_bound[2]

        mask = (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0)

        gt_depths = torch.where(
            mask,
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        if view_label.dim() == 5:
            C = other_labels.size(-1)
            other_labels = other_labels.view(B*N, H, W, C)
            other_labels = other_labels.view(B * N, H // downsample_factor,
                                   downsample_factor, W // downsample_factor,
                                   downsample_factor, C)
            other_labels = other_labels.permute(0, 1, 3, 5, 2, 4).contiguous()
            other_labels = other_labels.view(-1, downsample_factor * downsample_factor)
            nearst_idx = nearst_idx.unsqueeze(-1).repeat(1, C).view(-1)

            other_labels = other_labels.gather(1, nearst_idx.unsqueeze(-1))
            other_labels = other_labels.reshape(B * N, H // downsample_factor,
                                   W // downsample_factor, C)

            return gt_depths.float(), other_labels
        else:
            return gt_depths.float(), None

    @force_fp32()
    def loss(self,
             view_label,
             depth_preds,
             sem_preds=None,
             downsample_factor=None):

        depth_preds = depth_preds.flatten(0, 1)

        depth_label, other_label = self.get_downsampled_view_label(view_label, downsample_factor=downsample_factor)

        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_label, dim=1).values > 0.0
        depth_label = depth_label[fg_mask]
        depth_preds = depth_preds[fg_mask]
        depth_preds = depth_preds.softmax(-1)

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_label,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()) * self.depth_loss_weight

        if sem_preds is not None:
            sem_preds = sem_preds.flatten(0, 1)
            semantic_label = other_label[..., 0].view(-1).long()
            cls_channel = sem_preds.size(1)
            sem_preds = sem_preds.permute(0, 2, 3, 1).contiguous().view(
                -1, cls_channel)
            sem_preds = sem_preds[fg_mask]
            semantic_label = semantic_label[fg_mask]
            loss_cls = self.loss_cls(sem_preds, semantic_label)

            if self.with_lovasz_loss:
                loss_lovasz = self.loss_lovasz(sem_preds, semantic_label)
                loss_cls = loss_cls + loss_lovasz

            return depth_loss, loss_cls
        else:
            return depth_loss, None