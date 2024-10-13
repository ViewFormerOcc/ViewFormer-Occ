import torch
import numpy as np
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class ViewFormer(MVXTwoStageDetector):
    """ViewFormer."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=True,
                 use_temporal=True,
                 num_frame_backbone_grads=1,
                 num_frame_head_grads=1,
                 num_frame_losses=1,
                 depth_supvise=False,
                 depth_loss_weight=0.2,
                 with_view_semantic=True,
                 ):
        super(ViewFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.use_temporal = use_temporal

        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_losses = num_frame_losses

        self.prev_scene_token = None

        self.depth_supvise = depth_supvise
        # auxiliary depth task for backbone
        if self.depth_supvise:
            from projects.mmdet3d_plugin.bevdepth import Custom_DepthNet

            self.downsample_factor = [8, 16]
            self.with_view_semantic = with_view_semantic
            d_bound = [1.0, 52.0, 0.5]
            depth_channels = 108
            loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=depth_loss_weight * 2.0)
            loss_lovasz = dict(
                     type='LovaszLoss',
                     loss_weight=depth_loss_weight * 2.0)
            self.depth_net = Custom_DepthNet(
                256,
                256,
                depth_channels=depth_channels,
                with_semantics=with_view_semantic,
                d_bound=d_bound,
                depth_loss_weight=depth_loss_weight,
                loss_cls=loss_cls,
                with_lovasz_loss=True,
                loss_lovasz=loss_lovasz,
            )

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d=None,
                          gt_labels_3d=None,
                          gt_bboxes_ignore=None,
                          img_metas=None,
                          prev_exists=None,
                          voxel_semantics=None,
                          mask_lidar=None,
                          mask_camera=None,
                          requires_grad=True,
                          return_losses=False,
                          ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(pts_feats,
                                          img_metas,
                                          prev_exists=prev_exists,
                                          bev_only=not return_losses)
            self.train()
        else:
            outs = self.pts_bbox_head(pts_feats,
                                      img_metas,
                                      prev_exists=prev_exists,
                                      bev_only=not return_losses)

        if return_losses:
            loss_inputs = [voxel_semantics, mask_lidar, mask_camera, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

            return losses, outs['bev_embed']
        else:
            return None, outs['bev_embed']

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            num_inference_data = len(kwargs['img'])
            if num_inference_data == 1:
                return self.forward_test(**kwargs)
            else:
                return self.aug_test(**kwargs)

    def inference_img_feat(self, imgs_queue, per_frame=False, training=True):
        if training:
            self.eval()
        with torch.no_grad():
            batch_size, num_frame = imgs_queue.shape[:2]

            if per_frame:
                queue_img_feats = []
                for i in range(num_frame):
                    queue_img_feats.append(self.extract_feat(img=imgs_queue[:, i, ...]))
            else:
                img_feats = self.extract_feat(img=imgs_queue.flatten(0, 1))
                for lvl, lvl_feats in enumerate(img_feats):
                    img_feats[lvl] = lvl_feats.view(batch_size, num_frame, *lvl_feats.shape[1:])

                queue_img_feats = [[] for _ in range(num_frame)]
                for i in range(num_frame):
                    for lvl_feats in img_feats:
                        queue_img_feats[i].append(lvl_feats[:, i, ...])

        if training:
            self.train()
        return queue_img_feats

    def window_forward_tarin(self,
                      img_feats,
                      img_metas,
                      voxel_semantics=None,
                      mask_lidar=None,
                      mask_camera=None,
                      prev_exists=None):

        losses = dict()
        num_frame = len(img_feats)
        head_grad_start = num_frame - self.num_frame_head_grads
        loss_calc_start = num_frame - self.num_frame_losses

        num_label = voxel_semantics.size(1)
        assert self.num_frame_losses == num_label

        requires_grad = False
        return_losses = False
        label_idx = 0

        sequence_mode =  prev_exists is not None


        for i in range(num_frame):

            if i >= head_grad_start:
                requires_grad = True
            if i >= loss_calc_start:
                return_losses = True
                label_idx = i - loss_calc_start

            if not sequence_mode:
                batch_size = img_feats[0][0].size(0)
                if i == 0:
                    prev_exists = torch.zeros(batch_size, dtype=torch.bool).to(img_feats[0][0].device)
                else:
                    prev_exists = torch.ones(batch_size, dtype=torch.bool).to(img_feats[0][0].device)

            i_img_metas = [each[i] for each in img_metas]
            loss, space_embed = self.forward_pts_train(img_feats[i],
                                                       prev_exists=prev_exists,
                                                       img_metas=i_img_metas,
                                                       voxel_semantics=voxel_semantics[:, label_idx, ...],
                                                       mask_lidar=mask_lidar[:, label_idx, ...],
                                                       mask_camera=mask_camera[:, label_idx, ...],
                                                       requires_grad=requires_grad,
                                                       return_losses=return_losses)

            if loss is not None:
                if self.num_frame_losses > 1:
                    for key, value in loss.items():
                        losses[f'f{i}.{key}'] = value
                else:
                    losses = loss

        if not sequence_mode:
            self.clear_memory()

        return losses, space_embed


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      voxel_semantics=None,
                      mask_lidar=None,
                      mask_camera=None,
                      prev_exists=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        target_img = img[:, -self.num_frame_backbone_grads:, ...]
        temp_img = img[:, :-self.num_frame_backbone_grads, ...]

        target_img_feats = []
        for i in range(target_img.size(1)):
            mlvl_feats = self.extract_feat(target_img[:, i, ...])
            target_img_feats.append(mlvl_feats)

            if self.depth_supvise:
                this_img_metas = [each[-self.num_frame_backbone_grads + i] for each in img_metas]
                view_depth_loss_list, view_cls_loss_list = self.view_depth_sem_loss(mlvl_feats, this_img_metas)

        if self.use_temporal and temp_img.size(1) > 0:
            temporal_img_feats = self.inference_img_feat(temp_img, training=True)
            temporal_img_feats.extend(target_img_feats)
            img_feats = temporal_img_feats
        else:
            img_feats = target_img_feats

        losses, space_embed = self.window_forward_tarin(img_feats,
                                           img_metas=img_metas,
                                           voxel_semantics=voxel_semantics,
                                           mask_lidar=mask_lidar,
                                           mask_camera=mask_camera,
                                           prev_exists=prev_exists)

        if self.depth_supvise:
            count = 0
            for loss_view_depth_i, loss_view_cls_i in zip(view_depth_loss_list, view_cls_loss_list):
                losses[f'd{count}.loss_v_depth'] = loss_view_depth_i
                if self.with_view_semantic:
                    losses[f'd{count}.loss_v_cls'] = loss_view_cls_i
                count += 1

        return losses

    def view_depth_sem_loss(self, mlvl_feats, img_metas):
        view_depth_loss_list, view_cls_loss_list = [], []
        view_label = torch.stack([torch.stack(each['pixel_wise_label'], dim=0) for each in img_metas], dim=0)
        view_label = view_label.type_as(mlvl_feats[0])

        feats_for_depth = mlvl_feats[:len(self.downsample_factor)]
        for d_i, feats in enumerate(feats_for_depth):
            d_preds, sem_preds = self.depth_net(feats, img_metas)
            depth_loss, loss_cls = self.depth_net.loss(view_label, d_preds, sem_preds, self.downsample_factor[d_i])
            view_depth_loss_list.append(depth_loss)
            view_cls_loss_list.append(loss_cls)
        return view_depth_loss_list, view_cls_loss_list

    def forward_test(self, img_metas,
                     img=None,
                     voxel_semantics=None,
                     mask_lidar=None,
                     mask_camera=None,
                     **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        img = img[0]
        img_metas = img_metas[0]

        occ_results = self.simple_test(
            img_metas, img, **kwargs)

        return occ_results

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""

        if self.use_temporal:
            assert len(img_metas) == 1, 'multi batch sequnce inference is not supported yet'
            if img_metas[0]['scene_token'] != self.prev_scene_token:
                self.prev_scene_token = img_metas[0]['scene_token']
                prev_exists = torch.zeros(1, dtype=torch.bool).to(x[0].device)
            else:
                prev_exists = torch.ones(1, dtype=torch.bool).to(x[0].device)

        outs = self.pts_bbox_head(x, img_metas, prev_exists=prev_exists)

        occ_results = self.pts_bbox_head.get_results(
            outs, img_metas)

        return occ_results, outs['occ']

    def simple_test(self, img_metas, img, rescale=False):
        """Test function without augmentaiton."""

        img_feats = self.extract_feat(img)

        occ, _ = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        return occ