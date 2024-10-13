import os
from multiprocessing import Pool

import mmcv
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes

#from mmdet3d.core.bbox import LiDARInstance3DBoxes, get_box_type
#import torch
#from mmdet3d.ops.roiaware_pool3d import roiaware_pool3d_ext

def points_in_boxes_np(points, boxes):
    # note: boxes center is [0.5, 0.5, 0.5]

    rot = boxes[:, -1][:, None]
    rot = -rot
    cos, sin = np.cos(rot), np.sin(rot)
    zeros, ones = np.zeros_like(rot), np.ones_like(rot)
    rot_matrix = np.concatenate((cos, sin, zeros,
                        -sin, cos, zeros,
                        zeros, zeros, ones), axis=-1).reshape(-1, 3, 3)
    local_pints = points[None, :, :] - boxes[:, None, :3]
    local_pints = np.matmul(rot_matrix[:, None, :, :], local_pints[..., None])
    size = boxes[:, 3:6]
    in_box_matrix = (local_pints[:, :, 0] > -size[:, None, 0:1] / 2) & (local_pints[:, :, 0] < size[:, None, 0:1] / 2) \
        & (local_pints[:, :, 1] > -size[:, None, 1:2] / 2) & (local_pints[:, :, 1] < size[:, None, 1:2] / 2) \
        & (local_pints[:, :, 2] > -size[:, None, 2:3] / 2) & (local_pints[:, :, 2] < size[:, None, 2:3] / 2)

    in_box_mask = in_box_matrix[:, :].sum(0) > 0
    in_box_indices = in_box_matrix[:, :].argmax(0)
    in_box_indices[~in_box_mask] = -1

    return in_box_indices.reshape(-1)


# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, mask


data_root = 'data/nuscenes'
info_path = 'data/nuscenes/nuscenes_12hz_infos_val.pkl'


lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]

nusc = NuScenes(version='v1.0-trainval',
                dataroot=data_root,
                verbose=True)


def worker(info):
    lidar_path = info['lidar_infos'][lidar_key]['filename']
    points = np.fromfile(os.path.join(data_root, lidar_path),
                         dtype=np.float32,
                         count=-1).reshape(-1, 5)[..., :4]

    lidar_token = nusc.get('sample', info['sample_token'])['data']['LIDAR_TOP']
    pano_filepath = os.path.join(nusc.dataroot, nusc.get('panoptic', lidar_token)['filename'])
    seg_filepath = os.path.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])
    pts_panoptic_label = np.load(pano_filepath)['data']
    pts_semantic_label = np.fromfile(seg_filepath, dtype=np.uint8)
    assert pts_panoptic_label.shape[0] == pts_semantic_label.shape[0] == points.shape[0]

    pts_label = np.concatenate([pts_panoptic_label.reshape(-1, 1).astype(np.float32),
        pts_semantic_label.reshape(-1, 1).astype(np.float32)], axis=-1)

    lidar_calibrated_sensor = info['lidar_infos'][lidar_key][
        'calibrated_sensor']
    lidar_ego_pose = info['lidar_infos'][lidar_key]['ego_pose']
    for i, cam_key in enumerate(cam_keys):
        cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
        cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
        img = mmcv.imread(
            os.path.join(data_root, info['cam_infos'][cam_key]['filename']))
        pts_img, depth, mask = map_pointcloud_to_image(
            points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)

        pts_label_img = pts_label[mask, :]
        file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
        np.concatenate([pts_img[:2, :].T, depth[:, None], pts_label_img],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join(data_root, 'depth_pano_seg_gt',
                                        f'{file_name}.bin'))
    # plt.savefig(f"{sample_idx}")

if __name__ == '__main__':
    po = Pool(24)
    mmcv.mkdir_or_exist(os.path.join(data_root, 'depth_pano_seg_velo_gt'))
    infos = mmcv.load(info_path)

    for info_idx, info in enumerate(infos):
        info['info_idx'] = info_idx
        po.apply_async(func=worker, args=(info, ))
    po.close()
    po.join()
