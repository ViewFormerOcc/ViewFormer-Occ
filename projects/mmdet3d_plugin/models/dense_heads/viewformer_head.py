import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
import numpy as np

from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils import build_transformer

from projects.mmdet3d_plugin.models.utils.pos_encoding import pos2posembNd, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.viewformer_utils import custom_build_norm_layer, VoxelBevUNet, pillarscatter


@HEADS.register_module()
class ViewFormerHead(BaseModule):
    """Head of ViewFormer. 
    Args:
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 *args,
                 in_channels=256,
                 num_levels=4,
                 num_cams=6,
                 transformer=None,
                 sync_cls_avg_factor=True,
                 time_range=[0.0, 2.5],
                 loss_prob=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     ignore_index=0,
                     loss_weight=2.0),
                 loss_lovasz=dict(
                     type='LovaszLoss',
                     ignore_index=0,
                     loss_weight=2.0),
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 occ_pos_thr=0.35,
                 bev_h=100,
                 bev_w=100,
                 num_points_in_pillar=8,
                 use_mask_lidar=False,
                 use_mask_camera=False,
                 relative_relo_range=None,
                 with_position=True,
                 use_temporal=True,
                 num_memory=4,
                 bev_dim=126,
                 bev_feat_only=False,
                 space3D_net_cfg=dict(),
                 with_bev_vel=False,
                 loss_vel=None,
                 temporal_state_dim=2, # velocity
                 align_vel_by_semantic=False,
                 **kwargs):
        super(ViewFormerHead, self).__init__()

        num_classes = space3D_net_cfg['num_classes']
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is ViewFormerHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight


        self.num_classes = num_classes
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.in_channels = in_channels

        self.loss_prob = build_loss(loss_prob)
        self.loss_cls = build_loss(loss_cls)
        self.loss_lovasz = build_loss(loss_lovasz)
        if loss_vel is not None:
            self.loss_vel = build_loss(loss_vel)

        self.cls_out_channels = num_classes
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        self.use_temporal = use_temporal
        self.bev_feat_only = bev_feat_only

        self.space3D_net = Space3DNet(**space3D_net_cfg)

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_points_in_pillar = num_points_in_pillar # z-axis

        self.time_range = time_range
        self.use_mask_lidar = use_mask_lidar
        self.use_mask_camera = use_mask_camera
        self.relative_relo_range = relative_relo_range
        self.bev_dim = bev_dim

        # velocity related params
        self.with_bev_vel = with_bev_vel
        self.temporal_state_dim = temporal_state_dim
        self.align_vel_by_semantic = align_vel_by_semantic

        self.occ_pos_thr = occ_pos_thr

        self.num_memory = num_memory
        # memory queue
        self.memory_prev_info = {
            'prev_bev': [None for _ in range(num_memory)],
            'prev_state': [None for _ in range(num_memory)],
            'ego2global': [None for _ in range(num_memory)],
            'timestamp': [None for _ in range(num_memory)],
        }

        self.with_position = with_position
        if self.with_position:
            self.init_position_embedding()

        self.pc_range = nn.Parameter(torch.tensor(
            pc_range), requires_grad=False)
        self.relative_relo_range = nn.Parameter(torch.tensor(
            self.relative_relo_range), requires_grad=False)

        self._init_layers()


    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        self.query_embedding = nn.Sequential(
            nn.Linear(2*self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.input_proj = nn.ModuleList()
        for i in range(self.num_levels):
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.embed_dims, 1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, 1),
            ))

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, 1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, 1),
            )

            self.fpe = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, 1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, 1),
            )

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_levels, self.embed_dims))
        self.camera_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))

        if self.use_temporal:
            self.ego_trans_embedding = nn.Sequential(
                    nn.Linear(144, self.bev_dim),
                    nn.ReLU(),
                    nn.Linear(self.bev_dim, self.bev_dim),
                )
            self.time_embedding = nn.Sequential(
                    nn.Linear(self.bev_dim, self.bev_dim),
                    nn.LayerNorm(self.bev_dim)
                )

            self.temporal_bev_proj = nn.Sequential(
                nn.Conv2d(self.bev_dim, self.bev_dim, 1),
                custom_build_norm_layer(self.bev_dim, 'LN', in_format='channels_first', out_format='channels_first'),
                nn.GELU(),
                nn.Conv2d(self.bev_dim, self.bev_dim, 1),
            )

            self.temporal_pos_embedding = nn.Sequential(
                nn.Linear(self.bev_dim, self.bev_dim),
                nn.ReLU(),
                nn.Linear(self.bev_dim, self.bev_dim),
            )

            if self.with_bev_vel:
                self.vel_embedding = nn.Sequential(
                    nn.Linear(24, self.bev_dim),
                    nn.ReLU(),
                    nn.Linear(self.bev_dim, self.bev_dim),
                )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

        if hasattr(self, 'learned_z_position'):
            z_range = self.pc_range[5] - self.pc_range[2]
            zs = torch.linspace(0., z_range, self.num_points_in_pillar) / z_range
            zs = zs.unsqueeze(0).repeat(self.learned_z_position.weight.size(0), 1)
            self.learned_z_position.weight.data = zs

        if not self.bev_feat_only:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.space3D_net.prob_branch[-1].bias, bias_init)

        if hasattr(self, 'level_embeds'):
            normal_(self.level_embeds)
        if hasattr(self, 'camera_embeds'):
            normal_(self.camera_embeds)

    def init_position_embedding(self):
        # pos embedding 3D, refer to PETR
        depth_num = 64
        depth_start = 1.0
        self.position_dim = depth_num * 3
        self.position_range = nn.Parameter(torch.tensor(
            [-51.2, -51.2, -10.0, 51.2, 51.2, 10.0]), requires_grad=False)
        index  = torch.arange(start=0, end=depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - depth_start) / (depth_num * (1 + depth_num))
        coords_d = depth_start + bin_size * index * index_1
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

    def clear_memory(self, clear_index=None, img_metas=None):
        if clear_index is None:
            self.memory_prev_info.clear()
            self.memory_prev_info = {
                'prev_bev': [None for _ in range(self.num_memory)],
                'prev_state': [None for _ in range(self.num_memory)],
                'ego2global': [None for _ in range(self.num_memory)],
                'timestamp': [None for _ in range(self.num_memory)],
            }
        else:
            ego2global, timestamp = [], []
            for img_meta in img_metas:
                ego2global.append(img_meta['ego2global'])
                timestamp.append(img_meta['timestamp'])
            ego2global = np.asarray(ego2global, dtype=np.float64)
            timestamp = np.asarray(timestamp, dtype=np.float64)

            # to clear subbatch, we just set feat. to 0. and set all placeholder with current metas
            for i in range(self.num_memory):
                if self.memory_prev_info['prev_bev'][i] is not None:
                    self.memory_prev_info['prev_bev'][i][clear_index] = 0.
                    if self.memory_prev_info['prev_state'][i] is not None:
                        self.memory_prev_info['prev_state'][i][clear_index] = 0.
                    self.memory_prev_info['ego2global'][i][clear_index] = ego2global[clear_index]
                    self.memory_prev_info['timestamp'][i][clear_index] = timestamp[clear_index]
        return

    def update_memory(self, bev_embed, img_metas, state=None):
        ego2global, timestamp = [], []
        for img_meta in img_metas:
            ego2global.append(img_meta['ego2global'])
            timestamp.append(img_meta['timestamp'])
        ego2global = np.asarray(ego2global, dtype=np.float64)
        timestamp = np.asarray(timestamp, dtype=np.float64)

        self.memory_prev_info['prev_bev'].append(bev_embed.detach())
        self.memory_prev_info['prev_state'].append(state.detach() if state is not None else None)
        self.memory_prev_info['ego2global'].append(ego2global)
        self.memory_prev_info['timestamp'].append(timestamp)

        for key in self.memory_prev_info.keys():
            self.memory_prev_info[key].pop(0)

        return

    def gen_pesudo_placeholder(self, batch_size, tensor):
        embed_shape = [self.bev_dim] + self.space3D_net.out_shape[1:]
        if self.with_bev_vel:
            state_shape = [self.temporal_state_dim]+ self.space3D_net.out_shape[1:]
        pesudo_bev_embed = tensor.new_zeros(batch_size, *embed_shape)

        if self.with_bev_vel:
            pesudo_state = tensor.new_zeros(batch_size, *state_shape)
        else:
            pesudo_state = None

        return pesudo_bev_embed, pesudo_state

    def temporal_align_flow_matching(self, img_metas):
        temporal_bev = [each for each in self.memory_prev_info['prev_bev'] if each is not None]
        temporal_bev = torch.stack(temporal_bev, dim=1) if len(temporal_bev) > 0 else None
        if temporal_bev is None:
            return None, None, None, None, None

        if temporal_bev.dim() == 6:
            batch_size, num_frame, _, Z, H, W = temporal_bev.shape
        elif temporal_bev.dim() == 5:
            batch_size, num_frame, _, H, W = temporal_bev.shape
            Z = 1

        temporal_ego2global = [each for each in self.memory_prev_info['ego2global'] if each is not None]
        temporal_timestamp = [each for each in self.memory_prev_info['timestamp'] if each is not None]

        temporal_ego2global = np.asarray(temporal_ego2global, dtype=np.float64).transpose(1, 0, 2, 3) # (bs, num_frame, 4, 4)
        temporal_timestamp = np.asarray(temporal_timestamp, dtype=np.float64).transpose(1, 0)

        target_ego2global, target_timestamp = [], []
        for img_meta in img_metas:
            target_ego2global.append(img_meta['ego2global'])
            target_timestamp.append(img_meta['timestamp'])
        target_ego2global = np.asarray(target_ego2global, dtype=np.float64)[:, np.newaxis, ...]
        target_timestamp = np.asarray(target_timestamp, dtype=np.float64)[:, np.newaxis]

        target2temporal = np.matmul(np.linalg.inv(temporal_ego2global), target_ego2global)
        target2temporal = temporal_bev[0].new_tensor(target2temporal) # (num_frame, 4, 4)

        pc_range = self.pc_range

        # use ref_points (per cell) to realize temporal feat. warp
        ref_points = self.get_reference_points(H, W, Z, device=temporal_bev.device, dtype=temporal_bev.dtype)
        ref_points[..., 0] = ref_points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_points[..., 1] = ref_points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_points[..., 2] = ref_points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref_points = ref_points.view(1, 1, ref_points.size(0), ref_points.size(1), 3).repeat(batch_size, num_frame, 1, 1, 1)
        ref_points = torch.cat([ref_points, torch.ones_like(ref_points[..., :1])], dim=-1)

        temporal_ref_points = torch.matmul(target2temporal[:, :, None, None, :, :], ref_points.unsqueeze(-1)).squeeze(-1)
        temporal_ref_points = temporal_ref_points[..., :3]

        # norm to match grid_sample API
        temporal_ref_points[..., 0] = (temporal_ref_points[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        temporal_ref_points[..., 1] = (temporal_ref_points[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        temporal_ref_points[..., 2] = (temporal_ref_points[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
        temporal_ref_points = temporal_ref_points * 2.0 - 1.0

        if Z == 1:
            temporal_ref_points = temporal_ref_points.view(batch_size, num_frame, H, W, 3)[..., :2]
            # feat. warp to compensate ego motion
            aligned_temp_feat = F.grid_sample(
                temporal_bev.flatten(0, 1),
                temporal_ref_points.flatten(0, 1),
                align_corners=False).view(batch_size, num_frame, -1, H, W)
        else:
            temporal_ref_points = temporal_ref_points.view(batch_size, num_frame, Z, H, W, 3)
            aligned_temp_feat = F.grid_sample(
                temporal_bev.flatten(0, 1),
                temporal_ref_points.flatten(0, 1),
                align_corners=False).view(batch_size, num_frame, -1, Z, H, W)
            aligned_temp_feat = self.temporal_bev_encoder(aligned_temp_feat.permute(0, 1, 3, 4, 5, 2))
            aligned_temp_feat = aligned_temp_feat.max(dim=2)[0]
            aligned_temp_feat = aligned_temp_feat.permute(0, 1, 4, 2, 3) # batch_size, num_frame, -1, H, W

        with_velocity = self.with_bev_vel
        if with_velocity:
            temporal_vel = [each for each in self.memory_prev_info['prev_state'] if each is not None]
            temporal_vel = torch.stack(temporal_vel, dim=1)
            temporal_vel = F.grid_sample(
                temporal_vel.flatten(0, 1),
                temporal_ref_points.flatten(0, 1) if Z ==1 else temporal_ref_points[:, :, 0, ..., :2].flatten(0, 1),
                mode='nearest',
                align_corners=False).view(batch_size, num_frame, -1, H, W)

        temporal2target = np.matmul(np.linalg.inv(target_ego2global), temporal_ego2global)

        # generate vel embeds (value side)
        if with_velocity:

            # trans. temporal vel from temporal ego. to current ego.
            _temporal2target = torch.from_numpy(temporal2target).type_as(temporal_bev)
            temporal_vel = temporal_vel.permute(0, 1, 3, 4, 2)
            temporal_vel = torch.cat([temporal_vel, torch.zeros_like(temporal_vel[..., 0:1])], dim=-1)
            temporal_vel = torch.matmul(_temporal2target[:, :, None, None, :3, :3], temporal_vel.unsqueeze(-1)).squeeze(-1)[..., :2]
            temporal_vel = temporal_vel.view(batch_size, num_frame, H*W, 2)

            ego_dT_min, ego_dT_max = self.relative_relo_range[:2], self.relative_relo_range[3:5]
            temporal_vel_norm = (temporal_vel - ego_dT_min[None, None, None, :]) / (ego_dT_max - ego_dT_min)[None, None, None, :]
            motion_embeds = self.vel_embedding(nerf_positional_encoding(temporal_vel_norm))
            motion_embeds = motion_embeds.permute(0, 1, 3, 2).view(batch_size, num_frame, -1, H, W)
            aligned_temp_feat = aligned_temp_feat + motion_embeds

        query_vel, query_vel_embeds = None, None
        all_frame_dt = None

        relative_time = temporal_timestamp - target_timestamp

        relative_info = dict(
            timestamp=relative_time,
            trans_matrix=temporal2target,
        )

        return aligned_temp_feat, relative_info, query_vel, query_vel_embeds, all_frame_dt

    def position_embeding_3d(self, img_feats, img_metas):
        eps = 1e-5
        img_h, img_w, _ = img_metas[0]['img_shape'][0]
        B, N, C, H, W = img_feats.shape
        coords_h = torch.arange(H, device=img_feats.device).float() * img_h / H
        coords_w = torch.arange(W, device=img_feats.device).float() * img_w / W

        coords_d = self.coords_d.clone().type_as(coords_h)

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0).contiguous() # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        img2ego = []
        for img_meta in img_metas:
            ego2lidar = img_meta['ego2lidar']
            lidar2img = np.asarray(img_meta['lidar2img'])
            ego2img = np.matmul(lidar2img, ego2lidar[np.newaxis, :, :])
            img2ego.append(img_feats.new_tensor(np.linalg.inv(ego2img)))
        img2ego = torch.stack(img2ego, dim=0) # B, N, 4, 4

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2ego = img2ego.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2ego, coords).squeeze(-1)[..., :3]

        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        # refer to petr, petrv2
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        feats_drived_weight = self.fpe(img_feats.view(B*N, -1, H, W))
        feats_drived_weight = feats_drived_weight.sigmoid()

        coords_position_embeding = coords_position_embeding * feats_drived_weight

        return coords_position_embeding.view(B, N, -1, H, W)

    def temporal_feat_generation(self, img_metas):

        temporal_bev, relative_info, query_vel, query_vel_embeds, all_frame_dt = self.temporal_align_flow_matching(img_metas)
        if temporal_bev is None:
            return None, None

        batch_size, num_temporal = temporal_bev.shape[:2]
        if hasattr(self, 'temporal_bev_proj'):
            temporal_bev = self.temporal_bev_proj(temporal_bev.flatten(0, 1)).view(batch_size, num_temporal, -1, temporal_bev.size(-2), temporal_bev.size(-1))
        H, W = temporal_bev.shape[3:5]

        # generate pos_embed for temporal feat. based on relative time and ego trans. information
        timestamp = torch.from_numpy(relative_info['timestamp']).type_as(temporal_bev)
        trans_matrix = torch.from_numpy(relative_info['trans_matrix']).type_as(temporal_bev)

        time_range = self.time_range
        t_bound_min, t_bound_max = time_range[0], time_range[1]
        timestamp = (timestamp - t_bound_min) / (t_bound_max - t_bound_min)

        ego_dT_min, ego_dT_max = self.relative_relo_range[:3], self.relative_relo_range[3:]
        ego_trans_min = ego_dT_min * max(abs(t_bound_min), abs(t_bound_max))
        ego_trans_max = ego_dT_max * max(abs(t_bound_min), abs(t_bound_max))

        trans_matrix = trans_matrix[..., :3, :]
        # norm translation
        trans_matrix[..., :3, -1] = (trans_matrix[..., :3, -1] - ego_trans_min[None, None, :]) / (ego_trans_max - ego_trans_min)[None, None, :]
        trans_matrix = trans_matrix.flatten(2, 3)

        trans_embeds = nerf_positional_encoding(trans_matrix)
        trans_embeds = self.ego_trans_embedding(trans_embeds)

        time_embeds = pos2posembNd(timestamp.unsqueeze(-1).clamp(min=0., max=1.), num_pos_feats=self.bev_dim)
        time_embeds = self.time_embedding(time_embeds)
        time_embeds = time_embeds + trans_embeds

        pos_embed = time_embeds[..., None, None].repeat(1, 1, 1, H, W)

        reference_points = self.get_reference_points(
            H, W, 1, device=pos_embed.device, dtype=pos_embed.dtype).view(H, W, -1)[..., :2]
        position_embeds = self.temporal_pos_embedding(pos2posembNd(reference_points, num_pos_feats=self.bev_dim))
        pos_embed = pos_embed + position_embeds.permute(2, 0 ,1)[None, None, ...]

        # pad
        if num_temporal < self.num_memory:
            pad_feat = temporal_bev.new_zeros((batch_size, self.num_memory - num_temporal, *temporal_bev.shape[2:]))
            temporal_bev = torch.cat([temporal_bev, pad_feat], dim=1)
            pos_embed = torch.cat([pos_embed, pad_feat], dim=1)

            if all_frame_dt is not None:
                pad_dt = all_frame_dt.new_zeros((batch_size, self.num_memory - num_temporal))
                all_frame_dt = torch.cat([all_frame_dt, pad_dt], dim=-1)

        temporal_bev_list, pos_embed_list = [], []
        for i in range(temporal_bev.size(1)):
            temporal_bev_list.append(temporal_bev[:, i, ...])
            pos_embed_list.append(pos_embed[:, i, ...])

        level_pad_mask = torch.zeros(len(temporal_bev_list), dtype=torch.bool).to(temporal_bev.device)
        pad_start_idx = num_temporal
        level_pad_mask[pad_start_idx:] = True

        # mapping to transformer form
        spatial_shapes = []
        feat_flatten = []
        pos_embed_flatten = []
        for i in range(len(temporal_bev_list)):
            feat = temporal_bev_list[i]
            pos = pos_embed_list[i]
            b, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat.flatten(2).transpose(1, 2))
            pos_embed_flatten.append(pos.flatten(2).transpose(1, 2))


        feat_flatten = torch.cat(feat_flatten, 1)
        pos_embed = torch.cat(pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        temporal_dict = {
            'value': feat_flatten,
            'key_pos': pos_embed,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
            'level_pad_mask': level_pad_mask,
            'query_vel': query_vel,
            'query_vel_embeds': query_vel_embeds,
            'all_frame_dt': all_frame_dt,
        }

        return temporal_dict, temporal_bev

    def get_reference_points(self, H, W, Z=8, device='cuda', dtype=torch.float):
        if hasattr(self, 'learned_z_position'):
            zs = self.learned_z_position.weight.transpose(0, 1).view(Z, H, W)
        else:
            zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                                    device=device).view(-1, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.view(Z, -1, 3)
        return ref_3d

    def get_positional_embedding(self, mlvl_feats, img_metas):
        batch_size, num_cams = mlvl_feats[0].shape[:2]
        mlvl_coords_position_embeding = []
        for lvl, lvl_feats in enumerate(mlvl_feats):
            H, W = lvl_feats.shape[3:]

            if self.with_position:
                pos_3d_embed = self.position_embeding_3d(lvl_feats, img_metas)
            else:
                pos_3d_embed = 0

            pos_3d_embed += self.camera_embeds.view(1, self.num_cams, -1, 1, 1).repeat(
                batch_size, 1, 1, H, W)
            pos_3d_embed += self.level_embeds[lvl].view(1, 1, -1, 1, 1).repeat(
                batch_size, num_cams, 1, H, W)

            mlvl_coords_position_embeding.append(pos_3d_embed)
        return mlvl_coords_position_embeding

    def pre_memory_control(self, prev_exists, img_metas, batch_size=1, tensor=None):
        if prev_exists is not None and not prev_exists.all():
            if not prev_exists.any():
                # all clear
                self.clear_memory()
                if self.training:
                    # make a temporal placeholder for the training of the first frame in a sequence
                    pesudo_bev_embed, pesudo_state = self.gen_pesudo_placeholder(batch_size=batch_size, tensor=tensor)
                    self.update_memory(pesudo_bev_embed, img_metas, state=pesudo_state)
            else:
                # certain batch clear
                clear_batch_index = torch.nonzero(~prev_exists).squeeze()
                self.clear_memory(clear_batch_index, img_metas)
        return

    def forward(self,
                mlvl_feats,
                img_metas,
                prev_exists=None,
                bev_only=False):
        """Forward function.
        """

        self.pre_memory_control(prev_exists, img_metas, batch_size=mlvl_feats[0].size(0), tensor=mlvl_feats[0])

        batch_size, num_cams = mlvl_feats[0].shape[:2]
        for lvl, feats in enumerate(mlvl_feats):
            feats = self.input_proj[lvl](feats.flatten(0, 1))
            H, W = feats.shape[-2:]
            mlvl_feats[lvl] = feats.view(batch_size, num_cams, -1, H, W)

        pos_embed = self.get_positional_embedding(mlvl_feats, img_metas)

        reference_points = self.get_reference_points(
            self.bev_h, self.bev_w, self.num_points_in_pillar, device=mlvl_feats[0].device, dtype=mlvl_feats[0].dtype)

        reference_points = reference_points.view(-1, 3) # (Z*H*W, 3)
        query_embeds = self.query_embedding(pos2posembNd(reference_points, num_pos_feats=self.embed_dims * 2))

        reference_points = reference_points[None].repeat(batch_size, 1, 1)

        query_embeds = query_embeds[None].repeat(batch_size, 1, 1).permute(1, 0, 2)

        temporal_dict = None
        if self.use_temporal:
            # get aligned BEV temporal feat. from streaming memory queue
            temporal_dict, temporal_feat = self.temporal_feat_generation(img_metas)

        outs_query, outs_intermediate = self.transformer(
            mlvl_feats,
            query_embeds,
            pos_embed,
            reference_points=reference_points,
            img_metas=img_metas,
            data_dict=temporal_dict,
            decode_shape=[self.num_points_in_pillar, self.bev_h, self.bev_w],
            relative_relo_range=self.relative_relo_range,
            vel_embedding=self.vel_embedding if hasattr(self, 'vel_embedding') else None,
        )

        if outs_query.dim() == 3:
            outs_query = outs_query.unsqueeze(0)

        num_outs = outs_query.size(0)

        space_query = outs_query.permute(0, 2, 3, 1).reshape(num_outs, batch_size, self.embed_dims, self.num_points_in_pillar, self.bev_h, self.bev_w)

        outs = dict()

        # prediction head
        occ_out, space_embed, occ_vel, bev_vel, bev_feat = self.space3D_net(space_query[-1], bev_only=bev_only)
        memory_vel = None
        if self.with_bev_vel:
            outs['bev_vel'] = bev_vel.unsqueeze(0)

            memory_vel = bev_vel.clone()
            if self.align_vel_by_semantic:
                memory_vel = self.vel_post_process(memory_vel, occ_out)
            memory_vel = memory_vel.permute(0, 3, 2, 1) # (B, W, H, 2) -> (B, 2, H, W)

        outs['bev_embed'] = space_embed

        self.update_memory(outs['bev_embed'], img_metas, state=memory_vel)

        if self.bev_feat_only or bev_only:
            return outs

        outs['occ'] = occ_out.unsqueeze(0)

        return outs

    def vel_post_process(self, memory_vel, occ_out):
        # align memory vel by semantic (object mask), this func slightly imporves the flow task
        B, W, H, Z, C =  occ_out.shape
        occ_pred = occ_out.reshape(-1, C)
        occ_prob = occ_pred[:, 0].sigmoid()
        occ_sem_label = occ_pred[:, 1:].argmax(-1)
        occ_sem_label[occ_prob < self.occ_pos_thr] = self.num_classes-1
        occ_sem_label = occ_sem_label.view(B, W, H, Z)

        valid_vel_mask = (occ_sem_label > 0) & (occ_sem_label <= 10)
        valid_vel_mask = valid_vel_mask.any(-1)

        memory_vel[~valid_vel_mask] = 0.
        return memory_vel

    def loss_single(self,
                    voxel_semantics,
                    preds):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """

        prob_preds = preds[:, 0:1]
        cls_preds = preds[:, 1:]

        prob_label = voxel_semantics.clone().long()
        prob_label[prob_label < self.num_classes-1] = 0
        prob_label[prob_label == self.num_classes-1] = 1


        num_total_pos = prob_label[prob_label == 0].numel()
        num_prob_total_pos = prob_preds.new_tensor([num_total_pos])
        num_prob_total_pos = torch.clamp(reduce_mean(num_prob_total_pos), min=1).item()

        prob_weights = torch.ones_like(prob_label).float()
        loss_prob = self.loss_prob(
                prob_preds, prob_label, prob_weights, avg_factor=num_prob_total_pos)


        semantic_mask = voxel_semantics < self.num_classes-1
        cls_preds = cls_preds[semantic_mask]
        cls_label = voxel_semantics[semantic_mask]

        num_total_pos = cls_label.numel()

        cls_avg_factor = num_total_pos
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_preds, cls_label, avg_factor=cls_avg_factor)

        loss_lovasz = self.loss_lovasz(cls_preds, cls_label)

        return loss_prob, loss_cls, loss_lovasz

    def loss_vel_single(self,
                        voxel_vel,
                        preds):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """

        num_total_pos = voxel_vel.size(0)
        if self.sync_cls_avg_factor:
            num_total_pos = reduce_mean(
                preds.new_tensor([num_total_pos]))
        num_total_pos = max(num_total_pos, 1)

        loss_vel = self.loss_vel(preds, voxel_vel, avg_factor=num_total_pos)

        return loss_vel

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             #gt_bboxes_list,
             #gt_labels_list,
             voxel_semantics,
             mask_lidar,
             mask_camera,
             preds_dicts,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        assert ~(self.use_mask_lidar and self.use_mask_camera)
        if self.use_mask_lidar:
            mask = mask_lidar > 0
        else:
            mask = mask_camera > 0


        occ_preds = preds_dicts['occ']

        assert occ_preds.dim() == 6
        num_decoders = occ_preds.size(0)

        occ_list = [occ_preds[i][mask] for i in range(num_decoders)]
        voxel_semantics_list = [voxel_semantics[mask] for _ in range(num_decoders)]


        loss_dict = dict()
        loss_prob, loss_cls, loss_lovasz = multi_apply(self.loss_single, voxel_semantics_list, occ_list)
        loss_dict['loss_prob'] = loss_prob[-1]
        loss_dict['loss_cls'] = loss_cls[-1]
        loss_dict['loss_lovasz'] = loss_lovasz[-1]

        # for flow task
        if 'vel' in preds_dicts or 'bev_vel' in preds_dicts:
            gt_voxel_vel = []
            for img_meta in img_metas:
                gt_voxel_vel.append(img_meta['voxel_vel'].type_as(occ_preds))

            gt_vel = torch.stack(gt_voxel_vel, dim=0)
            valid_mask = (gt_vel[..., 0] > -100) & (gt_vel[..., 0] < 100)
            gt_vel[~valid_mask] = 0.
            if self.with_bev_vel:
                # select the normed max voxel vel in a pillar as the bev GT
                vel_norm = gt_vel.norm(dim=-1)
                _, max_idx = vel_norm.max(dim=-1)
                gt_bev_vel = gt_vel.gather(3, max_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, gt_vel.size(-1))).squeeze(3)

                # foreground
                bev_vel_mask = mask.any(-1) & valid_mask.any(-1)

                if bev_vel_mask.sum() > 0:
                    gt_bev_vel = gt_bev_vel[bev_vel_mask]
                    for i in range(preds_dicts['bev_vel'].size(0)):
                        vel_preds = preds_dicts['bev_vel'][i][bev_vel_mask]
                        loss_dict[f'loss_bev_vel_{i}'] = self.loss_vel_single(gt_bev_vel, vel_preds)
                else:
                    # trick for loss
                    bev_vel_mask[0, 0, 0] = True
                    gt_bev_vel = gt_bev_vel[bev_vel_mask]
                    for i in range(preds_dicts['bev_vel'].size(0)):
                        vel_preds = preds_dicts['bev_vel'][i][bev_vel_mask]
                        loss_dict[f'loss_bev_vel_{i}'] = self.loss_vel_single(gt_bev_vel, vel_preds) * 0.

        return loss_dict

    @force_fp32(apply_to=('preds'))
    def get_results(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        if preds_dicts['occ'].dim() == 5:
            occ_out = preds_dicts['occ'] # [bs, 200, 200, 16, 18]
        elif preds_dicts['occ'].dim() == 6:
            occ_out = preds_dicts['occ'][-1]

        B, W, H, Z, C =  occ_out.shape
        occ_out = occ_out.reshape(-1, C)
        occ_prob = occ_out[:, 0].sigmoid()
        occ_sem_label = occ_out[:, 1:].argmax(-1)

        occ_sem_label[occ_prob < self.occ_pos_thr] = self.num_classes-1

        occ_sem_label = occ_sem_label.view(B, W, H, Z)

        if 'vel' in preds_dicts or 'bev_vel' in preds_dicts:
            if 'vel' in preds_dicts:
                if preds_dicts['vel'].dim() == 5:
                    occ_vel = preds_dicts['vel'] # [bs, 200, 200, 16, 18]
                elif preds_dicts['vel'].dim() == 6:
                    occ_vel = preds_dicts['vel'][-1]
            else:
                occ_vel = preds_dicts['bev_vel'][-1]
                occ_vel = occ_vel.unsqueeze(-2).repeat(1, 1, 1, occ_sem_label.size(-1), 1)

            occ_sem_label = occ_sem_label.cpu().to(torch.uint8)
            occ_vel = occ_vel.cpu().to(torch.float16)

            out_dict = dict(
                occ_semantic=occ_sem_label,
                occ_velocity=occ_vel,
            )

            if 'bev_vel' in preds_dicts:
                out_dict['bev_velocity'] = preds_dicts['bev_vel'][-1].cpu().to(torch.float16)

            return out_dict

        occ_sem_label = occ_sem_label.cpu().to(torch.uint8)
        return occ_sem_label


class Space3DNet(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels=32,
                 bev_dim=126,
                 in_shape=[8, 100, 100],
                 out_shape=[16, 200, 200],
                 num_classes=18,
                 bev_feat_only=False,
                 bev_unet_backbone=dict(
                    type='UNET_CNN'),
                 with_bev_vel=False,
                 pc_range=None):
        super(Space3DNet, self).__init__()

        self.feat_channels = feat_channels
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.bev_feat_only = bev_feat_only
        self.with_bev_vel = with_bev_vel

        self.scale_list = [out_shape[0] // in_shape[0], out_shape[1] // in_shape[1], out_shape[2] // in_shape[2]]
        self.scale = np.prod(self.scale_list)
        if self.scale > 1:
            self.upsampler = nn.Sequential(
                nn.ConvTranspose3d(
                                in_channels,
                                feat_channels,
                                kernel_size=self.scale_list,
                                stride=self.scale_list,
                                padding=0,
                                bias=False),
                custom_build_norm_layer(feat_channels, 'LN', in_format='channels_first', out_format='channels_first'),
            )
        else:
            feat_channels = in_channels

        bev_unet_backbone["in_channels"] = bev_dim
        self.voxel_bev_unet = VoxelBevUNet(feat_channels, out_shape[0], bev_unet_backbone)

        if self.with_bev_vel:
            self.bev_vel_branch = nn.Sequential(
                    nn.Conv2d(bev_dim, bev_dim, 1),
                    nn.GELU(),
                    nn.Conv2d(bev_dim, 2, 1),
                )

        if not bev_feat_only:
            self.prob_branch = nn.Sequential(
                nn.Linear(feat_channels, feat_channels*2),
                nn.GELU(),
                nn.Linear(feat_channels*2, 1),
            )
            self.cls_branch = nn.Sequential(
                nn.Linear(feat_channels, feat_channels*2),
                nn.GELU(),
                nn.Linear(feat_channels*2,num_classes-1),
            )

    def forward(self, x, out_space3D=False, bev_only=False, reg_only=False):

        if hasattr(self, 'upsampler'):
            x = self.upsampler(x)
        x, bev_feat = self.voxel_bev_unet(x, need_bev_feat=True)

        space_embed = bev_feat
        if self.bev_feat_only or bev_only:
            return None, bev_feat

        outputs, vel3D, out_bev_vel = None, None, None
        if not reg_only:
            x = x.permute(0, 4, 3, 2, 1) # --> (B, W, H, Z, C)
            prob = self.prob_branch(x)
            cls = self.cls_branch(x)
            outputs = torch.cat([prob, cls], dim=-1)

        if self.with_bev_vel:
            bev_vel = self.bev_vel_branch(bev_feat)
            out_bev_vel = bev_vel.permute(0, 3, 2, 1) # -> (B, W, H, 2)

        return outputs, space_embed, vel3D, out_bev_vel, bev_feat