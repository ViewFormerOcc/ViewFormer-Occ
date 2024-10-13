import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 need_geo_iou=True,
                 data_type='occ3D',
                 ):
        if data_type == 'surroundOcc':
            self.class_names = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                                'driveable_surface', 'other_flat', 'sidewalk',
                                'terrain', 'manmade', 'vegetation','free']
        elif num_classes == 18 or data_type == 'occ3D':
            self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                                'driveable_surface', 'other_flat', 'sidewalk',
                                'terrain', 'manmade', 'vegetation','free']
        elif num_classes == 17 or data_type == 'OpenOcc':
            self.class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
                            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation', 'free']

        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.need_geo_iou = need_geo_iou

        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

        if self.need_geo_iou:
            self.geo_hist = np.zeros((2, 2))

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist


    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

        if self.need_geo_iou:
            geo_pred = masked_semantics_pred
            geo_pred[geo_pred != (self.num_classes-1)] = 0
            geo_pred[geo_pred == (self.num_classes-1)] = 1
            geo_gt = masked_semantics_gt
            geo_gt[geo_gt != (self.num_classes-1)] = 0
            geo_gt[geo_gt == (self.num_classes-1)] = 1
            _, _hist = self.compute_mIoU(geo_pred, geo_gt, 2)
            self.geo_hist += _hist

    def count_miou(self, logger=None):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        if logger is not None:
            logger.info(f'===> per class IoU of {self.cnt} samples:')
        else:
            print(f'===> per class IoU of {self.cnt} samples:')
        #for ind_class in range(self.num_classes-1):
        for ind_class in range(self.num_classes):
            if logger is not None:
                logger.info(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
            else:
                print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        if logger is not None:
            logger.info(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        else:
            print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        # return mIoU

        if self.need_geo_iou:
            geo_mIoU = self.per_class_iu(self.geo_hist)

            if logger is not None:
                logger.info(f'===> geo - IoU = ' + str(round(geo_mIoU[0] * 100, 2)))
            else:
                print(f'===> geo - IoU = ' + str(round(geo_mIoU[0] * 100, 2)))



class Metric_FScore():
    def __init__(self,

                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8



    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera ):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self, logger=None):
        base_color, attrs = 'red', ['bold', 'dark']
        if logger is not None:
            logger.info(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))
        else:
            print(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))


class Metric_AveError():
    def __init__(self,
                 value_range_list=[[-0.2, 0.2], [0.2, 1e3], [0.0, 1e3]],
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.value_range_list = value_range_list
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask

        self.cnt = 0

        self.sum_err = [0. for _ in range(len(value_range_list))]
        self.sum_count = [0. for _ in range(len(value_range_list))]

    def compute_ave_err(self, pred, label):
        if label.shape[0] == 0:
            return 0, 0

        #err = np.abs(pred - label).sum()
        err = np.linalg.norm(pred - label, axis=-1).sum()
        count = label.shape[0]

        return err, count


    def add_batch(self, vel_pred, vel_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_vel_gt = vel_gt[mask_camera]
            masked_vel_pred = vel_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_vel_gt = vel_gt[mask_lidar]
            masked_vel_pred = vel_pred[mask_lidar]
        else:
            masked_vel_gt = vel_gt
            masked_vel_pred = vel_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)

        for i, value_range in enumerate(self.value_range_list):
            vel_norm = np.linalg.norm(masked_vel_gt, axis=1)
            mask = (vel_norm >= value_range[0]) & (vel_norm < value_range[1])
            err, conut = self.compute_ave_err(masked_vel_pred[mask], masked_vel_gt[mask])
            self.sum_err[i] += err
            self.sum_count[i] += conut

    def count_ave_err(self, logger=None, log_prifix='occ'):

        for i in range(len(self.value_range_list)):
            ave_err = self.sum_err[i] / max(self.sum_count[i], 1)

            if logger is not None:
                logger.info(f'===> Ave {log_prifix} velocity l2 error of intervel [{self.value_range_list[i]}]: {round(ave_err, 3)}')
            else:
                print(f'===> Ave {log_prifix} velocity l2 error of intervel [{self.value_range_list[i]}]: {round(ave_err, 3)}')


