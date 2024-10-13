
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
import os
import torch

from mmdet3d.core.bbox.box_np_ops import points_in_rbbox
from mmdet3d.ops import points_in_boxes_all

@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                num_sweeps=5,
                to_float32=False, 
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                random_choice=False,
                sweeps_id=None,
                merge_sweep2key=False,
                ):

        self.num_sweeps = num_sweeps    
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.random_choice = random_choice
        self.sweeps_id = sweeps_id

        self.merge_sweep2key = merge_sweep2key

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """

        sweep_imgs_list = []
        if self.merge_sweep2key:
            imgs = results['img']
            #timestamp = results['timestamp']
            sweep_imgs_list.extend(imgs)
            #timestamp_list = []
            #timestamp_list.extend(timestamp)
        else:
            render_metas = dict(
                img=[],
                lidar2img=[],
                cam_intrinsic=[],
                lidar2cam=[]
            )

        if self.pad_empty_sweeps and len(results['render_frames']) == 0:
            for i in range(self.num_sweeps):
                sweep_imgs_list.extend(imgs)
                for j in range(len(results['img'])):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['cam_intrinsic'].append(np.copy(results['intrinsic'][j]))
                    results['lidar2cam'].append(np.copy(results['lidar2cam'][j]))
        else:
            choices = [i for i in range(len(results['render_frames'])) if results['render_frames'][i]['sweep_idx'] in self.sweeps_id]

            for idx in choices:
                sweep = results['render_frames'][idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['render_frames'][idx - 1]
                if self.merge_sweep2key:
                    results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)

                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                if self.merge_sweep2key:
                    sweep_imgs_list.extend(img)
                    #sweep_ts = [timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
                    #timestamp_list.extend(sweep_ts)
                    for sensor in self.sensors:
                        results['lidar2img'].append(sweep[sensor]['lidar2img'])
                        results['cam_intrinsic'].append(sweep[sensor]['intrinsic'])
                        results['lidar2cam'].append(sweep[sensor]['lidar2cam'])
                else:
                    render_metas['img'].append(img)
                    lidar2img = []
                    cam_intrinsic = []
                    lidar2cam = []
                    for sensor in self.sensors:
                        lidar2img.append(sweep[sensor]['lidar2img'])
                        cam_intrinsic.append(sweep[sensor]['intrinsic'])
                        lidar2cam.append(sweep[sensor]['lidar2cam'])
                    render_metas['lidar2img'].append(lidar2img)
                    render_metas['cam_intrinsic'].append(cam_intrinsic)
                    render_metas['lidar2cam'].append(lidar2cam)

        if self.merge_sweep2key:
            results['img'] = sweep_imgs_list
            #results['timestamp'] = timestamp_list
        else:
            results['render_metas'] = render_metas

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str



@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
            pc_range=None,
            data_type='occ3D',
        ):
        self.data_root = data_root
        self.pc_range = pc_range
        self.data_type = data_type

        assert self.data_type in ['occ3D', 'OpenOcc', 'surroundOcc']

    def __call__(self, results):

        if self.data_type == 'surroundOcc':
            occ_gt_path = os.path.join(self.data_root, results['occ_gt_path'])
            occ = np.load(occ_gt_path).astype(np.int64)
            occ_class = occ[:, -1]
            occ_class[occ_class == 0] = 255 # ignore

            W, H, Z = [200, 200, 16] # occ size
            occupancy_classes = 17
            gt_occupancy = np.ones(W*H*Z, dtype=np.uint8)*occupancy_classes
            occ_index = occ[:, 0] * H*Z + occ[:, 1] * Z + occ[:, 2] # (x, y, z) format
            gt_occupancy[occ_index] = occ_class

            semantics = gt_occupancy.reshape(200, 200, 16)
            # fake mask
            mask_camera = np.ones_like(semantics, dtype=np.uint8)
            mask_lidar = mask_camera
        elif self.data_type == 'OpenOcc' or ('lc_occ_gt_path' in results and results['lc_occ_gt_path'] is not None):
            lc_occ_gt_path = results['lc_occ_gt_path']
            voxel_num = 200 * 200 * 16
            occupancy_classes = 16
            use_mask = False
            occ_gt_sparse = np.load(lc_occ_gt_path)
            occ_index = occ_gt_sparse[:, 0]
            occ_class = occ_gt_sparse[:, 1] 
            gt_occupancy = np.ones(voxel_num, dtype=np.uint8)*occupancy_classes
            gt_occupancy[occ_index] = occ_class

            occ_path, occ_name = os.path.split(results['lc_occ_gt_path'])
            invalid_path = os.path.join(occ_path, occ_name.split('.')[0] + '_invalid.npy')
            occ_invalid_index = np.load(invalid_path)
            visible_mask = np.ones(voxel_num, dtype=np.uint8)
            # we dont use mask to compare baseline in OpenOcc
            if use_mask:
                visible_mask[occ_invalid_index] = 0

            semantics = gt_occupancy.reshape(16, 200, 200).transpose(2, 1, 0)
            # placeholder value
            mask_camera = visible_mask.reshape(16, 200, 200).transpose(2, 1, 0)
            mask_lidar = mask_camera
        else:
            occ_gt_path = results['occ_gt_path']
            occ_gt_path = os.path.join(self.data_root,occ_gt_path)

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera



        if self.pc_range is not None:
            W, H, Z = semantics.shape
            pc_range = self.pc_range

            coords_w = (torch.arange(W).float() + 0.5) / W
            coords_h = (torch.arange(H).float() + 0.5) / H
            coords_z = (torch.arange(Z).float() + 0.5) / Z
            coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_z])).permute(1, 2, 3, 0)
            coords[..., 0] = coords[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
            coords[..., 1] = coords[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
            coords[..., 2] = coords[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

            occ_points = coords
            # reference points of GT, which used by global aug. process
            results['occ_points'] = occ_points.view(-1, 3)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@PIPELINES.register_module()
class GetVelGTFromBox(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            voxel_vel_path=None,
            pad_value=-1000,
            sparse_vel=False,
            vel_dim=2,
        ):
        self.voxel_vel_path = voxel_vel_path
        self.pad_value = pad_value
        self.sparse_vel = sparse_vel
        self.vel_dim = vel_dim

    def __call__(self, results):
        # print(results.keys())

        if self.voxel_vel_path is not None:
            W, H, Z = results['voxel_semantics'].shape
            if self.sparse_vel:
                voxel_vel = np.ones((W*H*Z, 2)) * self.pad_value
                sparse_vel = np.fromfile(os.path.join(self.voxel_vel_path, results['sample_idx']+'.bin'), dtype=np.float16).reshape(-1, self.vel_dim)[:, :2]
                sparse_idx = np.fromfile(os.path.join(self.voxel_vel_path, results['sample_idx']+'_idx.bin'), dtype=np.int32).reshape(-1)
                voxel_vel[sparse_idx] = sparse_vel
                voxel_vel = voxel_vel.reshape(W, H, Z, 2)
            else:
                voxel_vel_file_path = os.path.join(self.voxel_vel_path, results['sample_idx']+'.bin')
                voxel_vel = np.fromfile(voxel_vel_file_path, dtype=np.float16).reshape(W, H, Z, 2)
            results['voxel_vel'] = torch.from_numpy(voxel_vel).float()
            return results
        else:
            raise NotImplementedError

        return results