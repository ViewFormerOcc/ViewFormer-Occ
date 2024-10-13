_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
relative_relo_range=[-18.84, -18.84, -1.05, 18.84, 18.84, 1.05] # [dx_min, dy_min, dz_min, dx_max, dy_max, dz_max]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

point_class_names = [
    'ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

num_gpus = 8
batch_size = 1
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 90


bev_h_ = 100
bev_w_ = 100
num_points_in_pillar = 8
space_in_shape = [num_points_in_pillar, bev_h_, bev_w_]
space_out_shape = [16, 200, 200]

num_cams = 6
num_levels = 3
final_dim = (256, 704)

embed_dims = 72
num_heads = 9

num_frame_losses = 1
use_temporal = True
queue_length = 0

video_test_mode = True
num_memory = 4
voxel2bev = True
bev_dim = 126 if voxel2bev else embed_dims

time_range = [-num_memory*0.5 - 0.3, 0.] # extend 0.3s to ensure bound

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='ViewFormer',
    use_grid_mask=True,
    video_test_mode=video_test_mode,
    use_temporal=use_temporal,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_head_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    depth_supvise=True,
    img_backbone=dict(
        #pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=num_levels,
        add_extra_convs='on_output',
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='ViewFormerHead',
        pc_range=point_cloud_range,
        num_levels=num_levels,
        final_dim=final_dim,
        in_channels=256,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_points_in_pillar=num_points_in_pillar,
        time_range=time_range,
        use_mask_lidar=False,
        use_mask_camera=True,
        use_temporal=use_temporal,
        num_memory=num_memory,
        bev_dim=bev_dim,
        relative_relo_range=relative_relo_range,
        out_space3D_feat=False,
        space3D_net_cfg=dict(
            in_channels=embed_dims,
            bev_dim=bev_dim,
            feat_channels=32,
            in_shape=space_in_shape,
            out_shape=space_out_shape,
            num_classes=18,
        ),
        transformer=dict(
            type='ViewFormerTransformer',
            decoder=dict(
                type='ViewFormerTransformerDecoder',
                num_layers=4,
                return_intermediate=True,
                transformerlayers=dict(
                    type='ViewFormerTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='StreamTemporalAttn',
                            pc_range=point_cloud_range,
                            num_levels=num_memory,
                            embed_dims=bev_dim,
                            num_heads=num_heads,
                            data_from_dict=True,
                            voxel2bev=voxel2bev,
                            voxel_dim=embed_dims,
                            num_points=4,
                        ),
                        dict(
                            type='ViewAttn',
                            pc_range=point_cloud_range,
                            with_ffn=True,
                            num_levels=num_levels,
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            num_points=1,
                        )
                    ],
                    operation_order=('cross_attn', 'cross_attn')
                    ))),
        loss_prob=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0 * 3),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_lovasz=dict(
            type='LovaszLoss',
            loss_weight=1.0),
        ),
    )


dataset_type = 'NuSceneOcc'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


ida_aug_conf = {
        "resize_lim": (0.386, 0.55),
        "final_dim": final_dim,
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile',data_root=data_root, pc_range=point_cloud_range),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),

    dict(type='CustomResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='CustomGlobalRotScaleTransImage',
            flip_hv_ratio=[0.5, 0.5],
            pc_range=point_cloud_range,
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         keys=[ 'img', 'voxel_semantics', 'mask_lidar', 'mask_camera', 'prev_exists'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 
                    'scene_token', 'can_bus', 'ego2lidar', 'prev_idx', 'next_idx',
                    'ego2global', 'timestamp', 'img_trans_dict', 'ego_trans_dict',
                    'cam_intrinsic', 'cam2ego', 'pixel_wise_label', # for depth aux
                    )
        )
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile',data_root=data_root),
    dict(type='CustomResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=[ 'img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 
                    'scene_token', 'can_bus', 'ego2lidar', 'prev_idx', 'next_idx',
                    'ego2global', 'timestamp', 'img_trans_dict', 'ego_trans_dict',
                    )
                )
        ])
]


data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        queue_length=queue_length,
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        queue_length=queue_length,
        video_test_mode=video_test_mode,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        queue_length=queue_length,
        video_test_mode=video_test_mode,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )




optimizer = dict(
    type='AdamW', 
    lr=2e-4, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.25),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-3,
    )


evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = 'ckpts/r50_256x705_depth_pretrain.pth'  # the same pretrain-weights as in fb-occ


