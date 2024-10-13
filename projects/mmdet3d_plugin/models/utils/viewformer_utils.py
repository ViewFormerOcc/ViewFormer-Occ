import torch
import torch.nn as nn
import torch.nn.functional as F
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


class VoxelBevUNet(nn.Module):
    def __init__(self, in_dim, height_grid_num, bev_unet_backbone):
        super(VoxelBevUNet, self).__init__()

        bev_dim = bev_unet_backbone['in_channels']

        self.pillar_encoder = nn.Sequential(
            nn.Linear(in_dim, bev_dim, bias=False),
            custom_build_norm_layer(bev_dim, 'LN'),
            custom_build_act_layer('GELU'),
            nn.Linear(bev_dim, bev_dim),
        )

        self.unet = builder.build_backbone(bev_unet_backbone)

        self.decoder = self.decoder = nn.Sequential(
            nn.Conv2d(bev_unet_backbone['in_channels'], in_dim * height_grid_num, 1, bias=False),
            custom_build_norm_layer(in_dim * height_grid_num, 'LN', in_format='channels_first', out_format='channels_first'),
        )

    def forward(self, x, need_bev_feat=False, temporal_feat=None):

        identity = x

        B, C, Z, H, W = x.shape
        voxel_feat = x.permute(0, 2, 3, 4, 1)
        bev_feat = self.pillar_encoder(voxel_feat).max(dim=1)[0]
        bev_feat = bev_feat.permute(0, 3, 1, 2)

        bev_feat = self.unet(bev_feat)

        out_feat = self.decoder(bev_feat).view(B, C, Z, H, W)

        out_feat += identity

        if need_bev_feat:
            return out_feat, bev_feat
        else:
            return out_feat


def pillarscatter(pillar_feat, coords, H, W):
    in_channels = pillar_feat.size(1)
    canvas = torch.zeros(
        in_channels,
        H*W,
        dtype=pillar_feat.dtype,
        device=pillar_feat.device)
    feat = pillar_feat.transpose(0, 1)

    indices = coords[:, 0] * W + coords[:, 1]
    indices = indices.type(torch.long)
    canvas[:, indices] = feat
    return canvas.view(in_channels, H, W)


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = custom_build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x