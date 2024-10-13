
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER)
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.runner.base_module import BaseModule, ModuleList

from mmdet.models.utils.builder import TRANSFORMER

import warnings

import torch.utils.checkpoint as cp

import copy

from mmdet.models.backbones.resnet import BasicBlock

from mmcv.runner import force_fp32, auto_fp16

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch, MultiScaleDeformableAttnFunction
from timm.models.layers import DropPath
from mmcv import ConfigDict

from mmcv.cnn import build_norm_layer

from projects.mmdet3d_plugin.models.utils.pos_encoding import pos2posembNd, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.viewformer_utils import custom_build_norm_layer, MLPLayer


def position_decode(pts_xyz, pc_range=None):
    pts_xyz = pts_xyz.clone()
    pts_xyz[..., 0:1] = pts_xyz[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    pts_xyz[..., 1:2] = pts_xyz[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    pts_xyz[..., 2:3] = pts_xyz[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    return pts_xyz


@TRANSFORMER.register_module()
class ViewFormerTransformer(BaseModule):
    """Implements the ViewFormerTransformer transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 decoder=None,
                 **kwargs):
        super(ViewFormerTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)

        self.embed_dims = self.decoder.embed_dims

        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        pass

    def init_weights(self):
        # follow the official DETR to init parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, ViewAttn) or isinstance(m, StreamTemporalAttn):
                m.init_weight()

        self._is_init = True

    def forward(self,
                x,
                query_embed,
                pos_embed,
                reference_points=None,
                query_embedding=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            x (Tensor): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bs = x[0].size(0)

        spatial_shapes = []
        feat_flatten = []
        pos_embed_flatten = []
        for lvl, feat in enumerate(x):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(3).transpose(2, 3)
            feat_flatten.append(feat)
            if pos_embed is not None:
                pos_embed_flatten.append(pos_embed[lvl].flatten(3).transpose(2, 3))

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        if pos_embed is not None:
            pos_embed = torch.cat(pos_embed_flatten, 2)

        # out_dec: [num_layers, num_query, bs, dim]
        target = torch.zeros_like(query_embed)
        out_dec = self.decoder(
            query=target,
            key=None,
            value=feat_flatten,
            key_pos=pos_embed,
            query_pos=query_embed,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
            query_embedding=query_embedding,
            **kwargs)

        return out_dec


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class ViewFormerTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False,**kwargs):
        super(ViewFormerTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                query_pos=None,
                reference_points=None,
                reg_branches=None,
                query_embedding=None,
                **kwargs):
        """Forward function for `MCDetr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        intermediate = []
        intermediate_outs = []


        for lid, layer in enumerate(self.layers):
            query = layer(query, *args, query_pos=query_pos, reference_points=reference_points, **kwargs)

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_outs) if len(intermediate_outs) > 0 else None

        return query, None


@ATTENTION.register_module()
class ViewAttn(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=3,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 drop=0.,
                 drop_path=0.1,
                 kernel_size=3,
                 dilation=1,
                 num_points_in_pillar=4,
                 layer_scale=1.0,
                 mlp_ratio=4.,
                 act_layer='GELU',
                 batch_first=False,
                 need_center_grid=True,
                 attn_norm_sigmoid=True,
                 with_ffn=True,
                 grid_range_scale=1.0,
                 offset_single_level=True,
                 num_points=4,
                 point_dim=3,
                 with_cp=False,
                 **kwargs):
        super(ViewAttn, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        self.pc_range = pc_range
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_points_in_pillar = num_points_in_pillar
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_cams = num_cams

        self.attn_norm_sigmoid = attn_norm_sigmoid
        self.with_ffn = with_ffn
        self.offset_single_level = offset_single_level
        self.point_dim = point_dim

        self.need_center_grid = (self.kernel_size % 2 == 0) & (self.dilation % 2 ==0) & need_center_grid
        self.per_ref_points = num_points

        self.grid_range_scale = nn.Parameter(grid_range_scale * torch.ones(1), requires_grad=True)

        per_query_points = self.per_ref_points

        if self.offset_single_level:
            self.offset = nn.Linear(
                embed_dims, num_heads * 1 * per_query_points * self.point_dim)
        else:
            self.offset = nn.Linear(
                embed_dims, num_heads * num_levels * per_query_points * self.point_dim)
        self.weight = nn.Linear(embed_dims,
            num_heads * num_cams * num_levels * per_query_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.norm1 = custom_build_norm_layer(embed_dims, 'LN')

        if self.with_ffn:
            self.mlp = MLPLayer(in_features=embed_dims,
                                hidden_features=int(embed_dims * mlp_ratio),
                                act_layer=act_layer,
                                drop=drop)
            self.norm2 = custom_build_norm_layer(embed_dims, 'LN')

        self.gamma1 = nn.Parameter(layer_scale * torch.ones(embed_dims),
                                    requires_grad=True)
        if self.with_ffn:
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(embed_dims),
                                    requires_grad=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.batch_first = batch_first
        self.use_checkpoint = with_cp

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""

        # similar init concept as in deformabel-detr
        grid = self.generate_dilation_grids(self.kernel_size, self.kernel_size, self.dilation, self.dilation, 'cpu')
        assert (grid.size(0) == self.num_heads) & (self.embed_dims % self.num_heads == 0)

        grid = grid.unsqueeze(1).repeat(1, self.per_ref_points, 1)

        for i in range(self.per_ref_points):
            grid[:, i, ...] *= (i + 1)
        grid /= self.per_ref_points
        self.grid = grid

        if self.point_dim == 3:
            self.grid = torch.cat([torch.zeros_like(self.grid[..., :1]), self.grid], dim=-1)

        constant_init(self.offset, 0., 0.)
        constant_init(self.weight, 0., 0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def generate_dilation_grids(self, kernel_h, kernel_w, dilation_w, dilation_h, device):
        x, y = torch.meshgrid(
            torch.linspace(
                -((dilation_w * (kernel_w - 1)) // 2),
                -((dilation_w * (kernel_w - 1)) // 2) +
                (kernel_w - 1) * dilation_w, kernel_w,
                dtype=torch.float32,
                device=device),
            torch.linspace(
                -((dilation_h * (kernel_h - 1)) // 2),
                -((dilation_h * (kernel_h - 1)) // 2) +
                (kernel_h - 1) * dilation_h, kernel_h,
                dtype=torch.float32,
                device=device))
        grid = torch.stack([x, y], -1).reshape(-1, 2)

        if self.need_center_grid:
            grid = torch.cat([grid, torch.zeros_like(grid[0:1, :])], dim=0)
        return grid

    @force_fp32()
    def point_sampling(self, reference_points,  img_metas):
        ego2lidar = []
        lidar2img = []
        for img_meta in img_metas:
            ego2lidar.append(img_meta['ego2lidar'])
            lidar2img.append(img_meta['lidar2img'])
        ego2lidar = np.asarray(ego2lidar)
        ego2lidar = reference_points.new_tensor(ego2lidar) # (B, 4, 4)
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
        ego2img = lidar2img @ ego2lidar[:, None, :, :]

        num_cam = ego2img.size(1)
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.unsqueeze(1).repeat(1, num_cam, 1, 1, 1, 1, 1)

        reference_points_cam = torch.matmul(ego2img.to(torch.float32)[:, :, None, None, None, None, :], reference_points.to(torch.float32).unsqueeze(-1)).squeeze(-1)

        eps = 1e-5
        proj_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        proj_mask = (proj_mask & (reference_points_cam[..., 1:2] > 0.0)
                     & (reference_points_cam[..., 1:2] < 1.0)
                     & (reference_points_cam[..., 0:1] < 1.0)
                     & (reference_points_cam[..., 0:1] > 0.0))

        return reference_points_cam, proj_mask.squeeze(-1)

    def tangent_plane_sampling(self, reference_points, offset, img_metas):
        grid = self.grid.type_as(reference_points) * self.grid_range_scale # trans to pc_range

        local_pts = grid[None, None, :, None, ...] + offset
        if self.point_dim == 2:
            local_pts = torch.cat([torch.zeros_like(local_pts[..., :1]), local_pts], dim=-1)

        reference_points = position_decode(reference_points, self.pc_range)
        # the query-specific view angle mentioned in paper
        azimuth_angle = torch.atan2(reference_points[..., 1], reference_points[..., 0])

        rot_sin, rot_cos = torch.sin(azimuth_angle), torch.cos(azimuth_angle)
        zeros, ones = torch.zeros_like(azimuth_angle), torch.ones_like(azimuth_angle)
        rot_matrix = torch.stack([rot_cos, -rot_sin, zeros,
                                  rot_sin,  rot_cos, zeros,
                                  zeros,    zeros,   ones], dim=-1).reshape(*reference_points.shape[:-1], 3, 3)

        local_pts = torch.matmul(rot_matrix[:, :, None, None, None, ...], local_pts.unsqueeze(-1)).squeeze(-1)
        reference_points = reference_points[:, :, None, None, None, :] + local_pts

        reference_points_cam, proj_mask = self.point_sampling(reference_points, img_metas)

        if self.offset_single_level:
            reference_points_cam = reference_points_cam.repeat(1, 1, 1, 1, self.num_levels, 1, 1)
            proj_mask = proj_mask.repeat(1, 1, 1, 1, self.num_levels, 1)

        return reference_points_cam, proj_mask

    def attention(self,
                  query,
                  value,
                  reference_points_cam,
                  weights,
                  proj_mask,
                  spatial_shapes,
                  level_start_index,
                  ):
        num_query = query.size(1)
        bs, num_cam, num_value = value.shape[:3]
        num_all_points = weights.size(-1)

        slots = torch.zeros_like(query)

        # (bs, num_query, num_head, num_cam, num_level, num_p) 
        # --> (bs, num_cam, num_query, num_head, num_level, num_p)
        weights = weights.permute(0, 3, 1, 2, 4, 5).contiguous()

        # save memory trick, similar as bevformer_occ
        indexes = [[] for _ in range(bs)]
        max_len = 0
        for i in range(bs):
            for j in range(num_cam):
                index_query_per_img = proj_mask[i, j].flatten(1).sum(-1).nonzero().squeeze(-1)
                indexes[i].append(index_query_per_img)
                max_len = max(max_len, index_query_per_img.numel())

        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_cam_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, self.num_heads, self.num_levels, num_all_points, 2])
        weights_rebatch = weights.new_zeros(
            [bs, self.num_cams, max_len, self.num_heads, self.num_levels, num_all_points])

        for i in range(bs):
            for j in range(num_cam):  
                index_query_per_img = indexes[i][j]
                curr_numel = index_query_per_img.numel()
                queries_rebatch[i, j, :curr_numel] = query[i, index_query_per_img]
                reference_points_cam_rebatch[i, j, :curr_numel] = reference_points_cam[i, j, index_query_per_img]
                weights_rebatch[i, j, :curr_numel] = weights[i, j, index_query_per_img]

        value = value.view(bs*num_cam, num_value, self.num_heads, -1)
        sampling_locations = reference_points_cam_rebatch.view(bs*num_cam, max_len, self.num_heads, self.num_levels, num_all_points, 2)
        attention_weights = weights_rebatch.reshape(bs*num_cam, max_len, self.num_heads, self.num_levels, num_all_points)

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction_fp32.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = output.view(bs, num_cam, max_len, -1)
        for i in range(bs):
            for j in range(num_cam):
                index_query_per_img = indexes[i][j]
                slots[i, index_query_per_img] += output[i, j, :len(index_query_per_img)]
        return slots

    def forward_layer(self,
                      query,
                      key,
                      value,
                      identity=None,
                      query_pos=None,
                      key_pos=None,
                      key_padding_mask=None,
                      reference_points=None,
                      spatial_shapes=None,
                      level_start_index=None,
                      img_metas=None,
                      **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            identity (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if query_pos is not None:
            query = query + query_pos

        if key_pos is not None:
            value = value + key_pos

        value = self.value_proj(value)

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()
        bs, num_cam, num_value, _ = value.size()

        value = value.view(bs, num_cam, num_value, self.num_heads, -1)

        per_query_points = self.per_ref_points
        sampling_offsets = self.offset(query).view(
            bs, num_query, self.num_heads, -1, per_query_points, self.point_dim)

        reference_points_cam, proj_mask = self.tangent_plane_sampling(reference_points, sampling_offsets, img_metas)

        weights = self.weight(query).view(bs, num_query, self.num_heads, self.num_cams, self.num_levels, per_query_points)
        weights = weights.masked_fill(~proj_mask.permute(0, 2, 3, 1, 4, 5), float("-inf"))
        if self.attn_norm_sigmoid:
            weights = weights.view(bs, num_query, self.num_heads, -1).sigmoid()
        else:
            weights = weights.view(bs, num_query, self.num_heads, -1).softmax(-1)
        weights = torch.nan_to_num(weights)
        weights = weights.view(bs, num_query, self.num_heads, self.num_cams, self.num_levels, per_query_points)

        output = self.attention(query,
                                value,
                                reference_points_cam,
                                weights,
                                proj_mask,
                                spatial_shapes,
                                level_start_index)
        output = self.output_proj(output)
        output = output.permute(1, 0, 2)
        return output

    def forward(self,
                query,
                key,
                value,
                identity=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_metas=None,
                **kwargs):
        if self.use_checkpoint and self.training:
            query = cp.checkpoint(
                        self.inner_forward,
                        query,
                        key,
                        value,
                        identity,
                        query_pos,
                        key_pos,
                        key_padding_mask,
                        reference_points,
                        spatial_shapes,
                        level_start_index,
                        img_metas)
        else:
            query = self.inner_forward(
                        query,
                        key,
                        value,
                        identity,
                        query_pos,
                        key_pos,
                        key_padding_mask,
                        reference_points,
                        spatial_shapes,
                        level_start_index,
                        img_metas,
                        **kwargs)
        return query

    def inner_forward(self,
                query,
                key,
                value,
                identity=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_metas=None,
                **kwargs):
        
        if identity is None:
            identity = query

        output = self.forward_layer(
                    query,
                    key,
                    value,
                    identity,
                    query_pos,
                    key_pos,
                    key_padding_mask,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    img_metas,
                    **kwargs
                )

        query = identity + self.drop_path(self.gamma1 * self.norm1(output))
        if self.with_ffn:
            query = query + self.drop_path(self.gamma2 * self.norm2(self.mlp(query)))

        return query


@ATTENTION.register_module()
class StreamTemporalAttn(BaseModule):
    """An attention module used in Detr3d. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=3,
                 im2col_step=64,
                 drop=0.,
                 drop_path=0.1,
                 kernel_size=3,
                 dilation=1,
                 layer_scale=1.0,
                 mlp_ratio=4.,
                 act_layer='GELU',
                 batch_first=False,
                 with_ffn=False,
                 data_from_dict=False,
                 num_points_in_pillar=8,
                 voxel2bev=False,
                 voxel_dim=64,
                 attn_norm_sigmoid=True,
                 grid_range_scale=1.0,
                 num_points=4,
                 pc_range=None,
                 use_pc_range=True,
                 with_cp=False,
                 **kwargs):
        super(StreamTemporalAttn, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.with_ffn = with_ffn
        self.data_from_dict = data_from_dict
        self.num_points_in_pillar = num_points_in_pillar

        self.voxel2bev = voxel2bev
        self.attn_norm_sigmoid = attn_norm_sigmoid

        self.grid_range_scale = grid_range_scale
        self.num_points = num_points
        self.pc_range = pc_range
        self.use_pc_range = use_pc_range

        if self.voxel2bev:
            self.bev_encoder = nn.Sequential(
                nn.Linear(voxel_dim, embed_dims, bias=False),
                custom_build_norm_layer(embed_dims, 'LN', in_format='channels_last', out_format='channels_last'),
                nn.GELU(),
                nn.Linear(embed_dims, embed_dims),
            )

        self.need_center_grid = (self.kernel_size % 2 == 0) & (self.dilation % 2 ==0)
        self.per_ref_points = num_points

        self.offset = nn.Linear(
            embed_dims, num_heads * num_levels * self.per_ref_points * 2)
        self.weight = nn.Linear(embed_dims,
            num_heads * num_levels * self.per_ref_points)

        if self.voxel2bev:
            self.output_proj = nn.Linear(embed_dims, voxel_dim * num_points_in_pillar)
        else:
            self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.value_proj = nn.Linear(embed_dims, embed_dims)

        tmp_voxel_dim = voxel_dim if self.voxel2bev else embed_dims

        self.norm1 = custom_build_norm_layer(tmp_voxel_dim, 'LN')
        if self.with_ffn:
            self.mlp = MLPLayer(in_features=tmp_voxel_dim,
                                hidden_features=int(tmp_voxel_dim * mlp_ratio),
                                act_layer=act_layer,
                                drop=drop)
            self.norm2 = custom_build_norm_layer(tmp_voxel_dim, 'LN')

        self.gamma1 = nn.Parameter(layer_scale * torch.ones(tmp_voxel_dim),
                                    requires_grad=True)
        if self.with_ffn:
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(tmp_voxel_dim),
                                    requires_grad=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.batch_first = batch_first
        self.use_checkpoint = with_cp

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""

        grid = self.generate_dilation_grids(self.kernel_size, self.kernel_size, self.dilation, self.dilation, 'cpu')
        assert (grid.size(0) == self.num_heads) & (self.embed_dims % self.num_heads == 0)

        grid = grid.unsqueeze(1).repeat(1, self.per_ref_points, 1)

        for i in range(self.per_ref_points):
            grid[:, i, ...] *= (i + 1)
        grid /= self.per_ref_points
        self.grid = grid

        constant_init(self.offset, 0., 0.)
        constant_init(self.weight, 0., 0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if hasattr(self, 'level_weight'):
            constant_init(self.level_weight, 0., 0.)
        self._is_init = True

    def generate_dilation_grids(self, kernel_h, kernel_w, dilation_w, dilation_h, device):
        x, y = torch.meshgrid(
            torch.linspace(
                -((dilation_w * (kernel_w - 1)) // 2),
                -((dilation_w * (kernel_w - 1)) // 2) +
                (kernel_w - 1) * dilation_w, kernel_w,
                dtype=torch.float32,
                device=device),
            torch.linspace(
                -((dilation_h * (kernel_h - 1)) // 2),
                -((dilation_h * (kernel_h - 1)) // 2) +
                (kernel_h - 1) * dilation_h, kernel_h,
                dtype=torch.float32,
                device=device))
        grid = torch.stack([x, y], -1).reshape(-1, 2)

        if self.need_center_grid:
            grid = torch.cat([grid, torch.zeros_like(grid[0:1, :])], dim=0)
        return grid

    def patch_sampling(self, reference_points, offset, spatial_shapes, img_metas):

        grid = self.grid.type_as(reference_points) * self.grid_range_scale # trans to pc_range

        local_pts = grid[None, None, :, None, ...] + offset

        if self.use_pc_range:
            range_scale = local_pts.new_tensor([self.pc_range[3] - self.pc_range[0], self.pc_range[4] - self.pc_range[1]])
            local_pts = local_pts / range_scale[None, None, None, None, None, :]
        else:
            local_pts = local_pts / spatial_shapes[None, None, None, :, None, :]

        sampling_locations = reference_points[:, :, None, None, None, :2] + local_pts

        return sampling_locations

    def forward_layer(self,
                      query,
                      key,
                      value,
                      identity=None,
                      query_pos=None,
                      key_pos=None,
                      key_padding_mask=None,
                      reference_points=None,
                      spatial_shapes=None,
                      level_start_index=None,
                      level_pad_mask=None,
                      query_vel=None,
                      query_vel_pos=None,
                      all_frame_dt=None,
                      img_metas=None):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            identity (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if query_pos is not None:
            query = query + query_pos

        if key_pos is not None:
            value = value + key_pos

        value = self.value_proj(value)

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        if self.voxel2bev:
            bs, _, voxel_dim = query.shape
            query = query.view(bs, self.num_points_in_pillar, -1, voxel_dim)
            query = self.bev_encoder(query).max(1)[0]
            reference_points = reference_points.view(bs, self.num_points_in_pillar, -1, 3)[:, 0, ...]

        bs, num_query, _ = query.size()
        bs, num_value, _ = value.size()

        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.offset(query).view(
            bs, num_query, self.num_heads, -1, self.per_ref_points, 2)

        sampling_locations = self.patch_sampling(reference_points, sampling_offsets, spatial_shapes, img_metas)

        weights = self.weight(query).view(bs, num_query, self.num_heads, self.num_levels, self.per_ref_points)

        if level_pad_mask is not None:
            if level_pad_mask.any():
                level_pad_mask = level_pad_mask.view(1, 1, 1, self.num_levels, 1).repeat(
                    bs, num_query, self.num_heads, 1, self.per_ref_points)
                weights = weights.masked_fill(level_pad_mask, float("-inf"))

        if self.attn_norm_sigmoid:
            weights = weights.view(bs, num_query, self.num_heads, -1).sigmoid()
        else:
            weights = weights.view(bs, num_query, self.num_heads, -1).softmax(-1)
        weights = torch.nan_to_num(weights) # TODO: check here
        attention_weights = weights.view(bs, num_query, self.num_heads, self.num_levels, self.per_ref_points)

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction_fp32.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if self.voxel2bev:
            # recover voxel-wise query
            output = output.view(bs, -1, voxel_dim, self.num_points_in_pillar).permute(0, 3, 1, 2).reshape(bs, -1, voxel_dim)

        output = output.permute(1, 0, 2)
        return output

    def forward(self,
                query,
                key,
                value,
                identity=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                data_dict=None,
                img_metas=None,
                **kwargs):

        level_pad_mask = None
        if self.data_from_dict:
            if data_dict is None:
                return query
            else:
                value = data_dict['value']
                key_pos = data_dict['key_pos']
                spatial_shapes = data_dict['spatial_shapes']
                level_start_index = data_dict['level_start_index']
                level_pad_mask= data_dict['level_pad_mask']

                query_vel = data_dict['query_vel']
                query_vel_pos = data_dict['query_vel_embeds']
                all_frame_dt = data_dict['all_frame_dt']

        if identity is None:
            identity = query

        output = self.forward_layer(
                    query,
                    key,
                    value,
                    identity,
                    query_pos,
                    key_pos,
                    key_padding_mask,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    level_pad_mask,
                    query_vel,
                    query_vel_pos,
                    all_frame_dt,
                    img_metas)

        query = identity + self.drop_path(self.gamma1 * self.norm1(output))
        if self.with_ffn:
            query = query + self.drop_path(self.gamma2 * self.norm2(self.mlp(query)))

        return query



@TRANSFORMER_LAYER.register_module()
class ViewFormerTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        super(ViewFormerTransformerLayer, self).__init__(init_cfg)

        self.batch_first = batch_first

        if isinstance(operation_order, str):
            operation_order = tuple((operation_order,))
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[-1].embed_dims

        num_ffns = operation_order.count('ffn')
        if num_ffns > 0:
            self.ffns = ModuleList()
            if isinstance(ffn_cfgs, dict):
                ffn_cfgs = ConfigDict(ffn_cfgs)
            if isinstance(ffn_cfgs, dict):
                ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
            assert len(ffn_cfgs) == num_ffns
            for ffn_index in range(num_ffns):
                if 'embed_dims' not in ffn_cfgs[ffn_index]:
                    ffn_cfgs['embed_dims'] = self.embed_dims
                else:
                    assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
                self.ffns.append(
                    build_feedforward_network(ffn_cfgs[ffn_index],
                                            dict(type='FFN')))
        num_norms = operation_order.count('norm')
        if num_norms > 0:
            self.norms = ModuleList()
            for _ in range(num_norms):
                self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query