import torch
import numpy as np
import os
from pandas import Timestamp
from tqdm import tqdm
from mmdet3d.datasets import NuScenesDataset
import mmcv
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
import random
from nuscenes.utils.geometry_utils import transform_matrix
from .occ_metrics import Metric_mIoU, Metric_FScore, Metric_AveError

import math

import pickle

@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    PointClassMapping = {
        'animal': 'ignore',
        'human.pedestrian.personal_mobility': 'ignore',
        'human.pedestrian.stroller': 'ignore',
        'human.pedestrian.wheelchair': 'ignore',
        'movable_object.debris': 'ignore',
        'movable_object.pushable_pullable': 'ignore',
        'static_object.bicycle_rack': 'ignore',
        'vehicle.emergency.ambulance': 'ignore',
        'vehicle.emergency.police': 'ignore',
        'noise': 'ignore',
        'static.other': 'ignore',
        'vehicle.ego': 'ignore',
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
        'flat.driveable_surface': 'driveable_surface',
        'flat.other': 'other_flat',
        'flat.sidewalk': 'sidewalk',
        'flat.terrain': 'terrain',
        'static.manmade': 'manmade',
        'static.vegetation': 'vegetation'
    }

    POINT_CLASS_GENERAL = ('noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child',
                           'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
                           'human.pedestrian.police_officer', 'human.pedestrian.stroller',
                           'human.pedestrian.wheelchair', 'movable_object.barrier',
                           'movable_object.debris', 'movable_object.pushable_pullable',
                           'movable_object.trafficcone', 'static_object.bicycle_rack',
                           'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid',
                           'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
                           'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer',
                           'vehicle.truck', 'flat.driveable_surface', 'flat.other', 'flat.sidewalk',
                           'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation', 'vehicle.ego')

    POINT_CLASS_SEG = ('ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                       'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                       'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                       'vegetation')

    THING_CLASSES = ('barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck')

    POINT_LABEL_MAPPTING = []
    for name in POINT_CLASS_GENERAL:
        POINT_LABEL_MAPPTING.append(POINT_CLASS_SEG.index(PointClassMapping[name]))
    POINT_LABEL_MAPPTING = np.array(POINT_LABEL_MAPPTING, dtype=np.int32)

    def __init__(self,
                 queue_length=4,
                 seq_mode=False,
                 seq_split_num=1,
                 num_frame_losses=1,
                 video_test_mode=True,
                 eval_fscore=False,
                 eval_vel=False,
                 eval_bev_vel=False,
                 voxel_vel_path=None,
                 use_lidar_coord=False,
                 sparse_vel=False,
                 vel_dim=2,
                 data_type=None,
                 anno_file_path=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_fscore = eval_fscore
        self.eval_vel = eval_vel
        self.eval_bev_vel = eval_bev_vel
        self.queue_length = queue_length
        self.num_frame_losses = num_frame_losses
        self.video_test_mode = video_test_mode
        self.seq_mode = seq_mode
        self.use_lidar_coord = use_lidar_coord
        self.data_type = data_type
        self.anno_file_path = anno_file_path

        # when self.data_type is None, its the occ3D dataset accually
        #assert self.data_type in ['occ3D', 'OpenOcc', 'surroundOcc']

        self.voxel_vel_path = voxel_vel_path
        self.sparse_vel = sparse_vel
        self.vel_dim = vel_dim

        self.data_infos = self.load_annotations(self.ann_file)

        # refer to streampetr
        if self.seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 0
            self.seq_split_num = seq_split_num
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


    def load_annotations(self, ann_file, dataset_ratio=None):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if not isinstance(ann_file, str):
            ann_file = self.ann_file

        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]

        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        # we only use streaming video training for now
        if self.queue_length != 0:
            raise NotImplementedError

        queue = []
        index_list = list(range(index - self.queue_length, index))
        index_list.append(index)
        index_list.sort()

        # get target frame info and aug matrix, we aply the same aug matrix to other fames in window
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        target_example = self.pipeline(input_dict)
        queue.append(target_example)

        info = self.union2target(queue, index_list.index(index))

        return info

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def union2target(self, queue, target_frame_idx):

        target_info = queue[target_frame_idx]
        if not self.test_mode:
            imgs_list = []
            metas_map = []
            label_dict = dict(
                voxel_semantics=[],
                mask_lidar=[],
                mask_camera=[],
            )
            target_scene_token = target_info['img_metas'].data['scene_token']
            for i, each in enumerate(queue):
                if each['img_metas'].data['scene_token'] != target_scene_token:
                    # pad target frame info
                    frame_info = target_info
                else:
                    frame_info = each

                imgs_list.append(frame_info['img'].data)
                metas_map.append(frame_info['img_metas'].data)

                if i >= (len(queue) - self.num_frame_losses):
                    for key in label_dict.keys():
                        label_dict[key].append(torch.from_numpy(frame_info[key]))

            target_info['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
            target_info['img_metas'] = DC(metas_map, cpu_only=True)

            for key in label_dict.keys():
                target_info[key] = DC(torch.stack(label_dict[key]), cpu_only=False, stack=True)

        else:
            num_aug_data = len(target_info['img'])
            aug_imgs_list = []
            aug_metas_map = []
            for i in range(num_aug_data):
                imgs_list = []
                metas_map = []
                target_scene_token = target_info['img_metas'][i].data['scene_token']
                for j, each in enumerate(queue):
                    if each['img_metas'][i].data['scene_token'] != target_scene_token:
                        # pad target frame info
                        imgs_list.append(target_info['img'][i].data)
                        metas_map.append(target_info['img_metas'][i].data)
                    else:
                        imgs_list.append(each['img'][i].data)
                        metas_map.append(each['img_metas'][i].data)
                aug_imgs_list.append(DC(torch.stack(imgs_list), cpu_only=False, stack=True))
                aug_metas_map.append(DC(metas_map, cpu_only=True))
            target_info['img'] = aug_imgs_list
            target_info['img_metas'] = aug_metas_map

        return target_info

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        # standard protocal modified from SECOND.Pytorch
        occ_gt_path = info['occ_gt_path'] if 'occ_gt_path' in info else None
        lc_occ_gt_path = info['lc_occ_gt_path'] if 'lc_occ_gt_path' in info else None
        input_dict = dict(
            occ_gt_path=occ_gt_path,
            lc_occ_gt_path=lc_occ_gt_path,
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            scene_token=info['scene_token'],
            timestamp=info['timestamp'] / 1e6,
            render_frames=info['render_frames'] if 'render_frames' in info else None,
        )

        if self.data_type == 'surroundOcc':
            # we dont use surroundOcc pkl, trick to support surroundOcc annotation
            lidar_file_name = os.path.split(info['lidar_path'])[1]
            occ_gt_path = os.path.join(self.anno_file_path, lidar_file_name+'.npy')
            input_dict['occ_gt_path'] = occ_gt_path


        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        input_dict['ego2lidar'] = ego2lidar

        ego2global_rotation = info['ego2global_rotation']
        ego2global_translation = info['ego2global_translation']
        ego2global = transform_matrix(translation=ego2global_translation, rotation=Quaternion(ego2global_rotation),
                                     inverse=False)
        input_dict['ego2global'] = ego2global


        if self.use_lidar_coord:
            # trick to support lidar coordinate annotations, such as OpenOcc, surroundOcc
            # cause our code use the key 'ego2...' for transformation, we just change the value as 'lidar2...', but keep the key
            lidar2global = ego2global @ np.linalg.inv(ego2lidar)
            input_dict['ego2global'] = lidar2global
            input_dict['ego2lidar'] = np.eye(4)


        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam2ego_rts = []

            pixel_wise_label = []
            img_semantic = []

            for cam_type, cam_info in info['cams'].items():
                data_path = cam_info['data_path']
                image_paths.append(data_path)
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

                tmp_lidar2cam_rt = lidar2cam_rt.T
                cam2ego_rt = np.linalg.inv(ego2lidar) @ np.linalg.inv(tmp_lidar2cam_rt)
                cam2ego_rts.append(cam2ego_rt)

                # only for auxiliary depth loss training
                try:
                    # we use image-wise depth supervision only when we compare with FB-Occ in occ3D dataset
                    _, file_name = os.path.split(cam_info['data_path'])
                    view_point_label = np.fromfile(os.path.join(
                            self.data_root, 'depth_pano_seg_gt', f'{file_name}.bin'),
                                                dtype=np.float32,
                                                count=-1).reshape(-1, 5)

                    cam_gt_depth = view_point_label[:, :3]
                    cam_gt_pano = view_point_label[:, 3:4].astype(np.int32)
                    cam_sem_mask = self.POINT_LABEL_MAPPTING[cam_gt_pano // 1000]

                    pixel_wise_label.append(np.concatenate([
                        cam_gt_depth,
                        cam_sem_mask.astype(np.float32)], axis=-1))

                except:
                    pass


                if not self.test_mode: # for seq_mode
                    prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
                else:
                    prev_exists = None

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    cam2ego=cam2ego_rts,
                    pixel_wise_label=pixel_wise_label,
                    prev_exists=prev_exists,
                ))


        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            if self.video_test_mode:
                return self.prepare_test_data(idx)
            else:
                # for testing
                return self.prepare_train_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate_miou(self, occ_results, runner=None, logger=None, show_dir=None, **eval_kwargs):
        if show_dir is not None:
            if not os.path.exists(show_dir):
                os.mkdir(show_dir)
            print('\nSaving output and gt in {} for visualization.'.format(show_dir))
            begin=eval_kwargs.get('begin',None)
            end=eval_kwargs.get('end',None)
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18 if not self.use_lidar_coord else 17, # please use data_type
            use_lidar_mask=False,
            use_image_mask=True,
            data_type=self.data_type)
        if self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )

        if self.eval_vel:
            self.vel_eval_metrics = Metric_AveError(
                value_range_list=[[-0.2, 0.2], [0.2, 1e3], [0.0, 1e3]],
                use_lidar_mask=False,
                use_image_mask=True,
            )

        if self.eval_bev_vel:
            self.bev_vel_eval_metrics = Metric_AveError(
                value_range_list=[[-0.2, 0.2], [0.2, 1e3], [0.0, 1e3]],
                use_lidar_mask=False,
                use_image_mask=True,
            )

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            if self.eval_vel or self.eval_bev_vel:
                occ_vel = occ_pred['occ_velocity'] if 'occ_velocity' in occ_pred else None
                bev_vel = occ_pred['bev_velocity'] if 'bev_velocity' in occ_pred else None
                occ_pred = occ_pred['occ_semantic']
            elif isinstance(occ_pred, dict):
                occ_pred = occ_pred['occ_semantic']

            info = self.data_infos[index]

            if self.data_type == 'surroundOcc':

                lidar_file_name = os.path.split(info['lidar_path'])[1]
                occ_gt_path = os.path.join(self.data_root, self.anno_file_path, lidar_file_name+'.npy')
                occ = np.load(occ_gt_path).astype(np.int64)
                occ_class = occ[:, -1]
                occ_class[occ_class == 0] = 255 # ignore, surroundOcc baseline ignore label 0

                W, H, Z = [200, 200, 16] # occ size
                occupancy_classes = 17
                gt_occupancy = np.ones(W*H*Z, dtype=np.uint8)*occupancy_classes
                occ_index = occ[:, 0] * H*Z + occ[:, 1] * Z + occ[:, 2] # (x, y, z) format
                gt_occupancy[occ_index] = occ_class

                gt_semantics = gt_occupancy.reshape(200, 200, 16)
                mask_camera = np.ones_like(gt_semantics, dtype=bool)
                mask_camera[gt_semantics == 255] = False # work around, use mask to achieve ignore
                mask_lidar = mask_camera

                # after ignore 0 in gt and pred, the class start from 1, we change it from 0
                # because mIoU calculate fn start from class 0
                gt_semantics = gt_semantics - 1
                occ_pred = occ_pred - 1

            elif self.data_type == 'OpenOcc' or 'lc_occ_gt_path' in info:
                # trick to support OpenOcc
                lc_occ_gt_path = info['lc_occ_gt_path']
                voxel_num = 200 * 200 * 16
                occupancy_classes = 16
                occ_gt_sparse = np.load(lc_occ_gt_path)
                occ_index = occ_gt_sparse[:, 0]
                occ_class = occ_gt_sparse[:, 1] 
                gt_occupancy = np.ones(voxel_num, dtype=np.uint8)*occupancy_classes
                gt_occupancy[occ_index] = occ_class

                occ_path, occ_name = os.path.split(info['lc_occ_gt_path'])
                invalid_path = os.path.join(occ_path, occ_name.split('.')[0] + '_invalid.npy')
                occ_invalid_index = np.load(invalid_path) # OccNet baseline use invalid_index in evaluation
                visible_mask = np.ones(voxel_num, dtype=bool)
                visible_mask[occ_invalid_index] = False

                gt_semantics = gt_occupancy.reshape(16, 200, 200).transpose(2, 1, 0)
                mask_camera = visible_mask.reshape(16, 200, 200).transpose(2, 1, 0)
                mask_lidar = mask_camera
            else:
                occ_gt = np.load(os.path.join(self.data_root, info['occ_gt_path']))
                if show_dir is not None:
                    if begin is not None and end is not None:
                        if index>= begin and index<end:
                            sample_token = info['token']
                            save_path = os.path.join(show_dir,str(index).zfill(4))
                            np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)
                    else:
                        sample_token=info['token']
                        save_path=os.path.join(show_dir,str(index).zfill(4))
                        np.savez_compressed(save_path,pred=occ_pred,gt=occ_gt,sample_token=sample_token)


                gt_semantics = occ_gt['semantics']
                mask_lidar = occ_gt['mask_lidar'].astype(bool)
                mask_camera = occ_gt['mask_camera'].astype(bool)
            occ_pred = occ_pred.squeeze(dim=0).cpu().numpy().astype(np.uint8)

            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            if self.eval_vel or self.eval_bev_vel:
                # work around
                assert self.voxel_vel_path is not None
                if self.sparse_vel:
                    W, H, Z = gt_semantics.shape
                    voxel_vel = np.ones((W*H*Z, 2)) * -1000
                    sparse_vel = np.fromfile(os.path.join(self.voxel_vel_path, info['token']+'.bin'), dtype=np.float16).reshape(-1, self.vel_dim)[:, :2]
                    sparse_idx = np.fromfile(os.path.join(self.voxel_vel_path, info['token']+'_idx.bin'), dtype=np.int32).reshape(-1)
                    voxel_vel[sparse_idx] = sparse_vel
                    gt_voxel_vel = voxel_vel.reshape(W, H, Z, 2)
                else:
                    voxel_vel_file_path = os.path.join(self.voxel_vel_path, info['token']+'.bin')
                    gt_voxel_vel = np.fromfile(voxel_vel_file_path, dtype=np.float16).reshape(*gt_semantics.shape, 2)
                gt_voxel_vel = gt_voxel_vel.astype(np.float32)
                valid_mask = gt_voxel_vel[..., 0] != -1000

                # flow is a TP error
                thing_mask = (occ_pred > 0) & (occ_pred <= 10) & (gt_semantics > 0) & (gt_semantics <= 10)
                valid_mask = valid_mask & thing_mask

                if self.eval_vel:
                    occ_vel = occ_vel.squeeze(dim=0).numpy()
                    occ_vel = occ_vel.astype(np.float32)
                    self.vel_eval_metrics.add_batch(occ_vel, gt_voxel_vel, mask_lidar & valid_mask, mask_camera & valid_mask)
                if self.eval_bev_vel:
                    bev_vel = bev_vel.squeeze(dim=0).numpy()
                    bev_vel = bev_vel.astype(np.float32)

                    gt_vel = torch.from_numpy(gt_voxel_vel)
                    gt_vel[~valid_mask] = 0.
                    vel_norm = gt_vel.norm(dim=-1)
                    _, max_idx = torch.max(vel_norm, dim=2)
                    gt_vel = gt_vel.gather(2, max_idx.unsqueeze(-1).unsqueeze(-1).repeat( 1, 1, 1, gt_vel.size(-1))).squeeze(2)
                    gt_vel = gt_vel.numpy()
                    self.bev_vel_eval_metrics.add_batch(bev_vel, gt_vel,
                                                        mask_lidar.any(axis=-1) & valid_mask.any(axis=-1),
                                                        mask_camera.any(axis=-1) & valid_mask.any(axis=-1))

        if logger is None and runner is not None:
            logger = runner.logger

        self.occ_eval_metrics.count_miou(logger=logger)
        if self.eval_fscore:
            self.fscore_eval_metrics.count_fscore(logger=logger)
        if self.eval_vel:
            self.vel_eval_metrics.count_ave_err(logger=logger, log_prifix='occ')
        if self.eval_bev_vel:
            self.bev_vel_eval_metrics.count_ave_err(logger=logger, log_prifix='bev')

    def format_results(self, occ_results,submission_prefix,**kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix,'{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.squeeze(dim=0).numpy().astype(np.uint8))
        print('\nFinished.')


