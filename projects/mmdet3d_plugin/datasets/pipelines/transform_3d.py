import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES

from PIL import Image
import torch
import copy


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        #results['img'] = padded_img
        #results['img_shape'] = [img.shape for img in padded_img]
        results['img_shape'] = [img.shape for img in results['img']]
        results['pad_shape'] = [img.shape for img in padded_img]    
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

        results['img'] = padded_img

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class CropMultiViewImage(object):
    """Crop the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, size=None):
        self.size = size

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        results['img'] = [img[:self.size[0], :self.size[1], ...] for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img_fixed_size'] = self.size
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        return repr_str


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        np.random.shuffle(self.scales)
        rand_scale = self.scales[0]
        img_shape = results['img_shape'][0]
        y_size = int(img_shape[0] * rand_scale)
        x_size = int(img_shape[1] * rand_scale) 
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size, y_size), return_scale=False) for img in results['img']]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['gt_bboxes_3d'].tensor[:, :6] *= rand_scale
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class HorizontalRandomFlipMultiViewImage(object):

    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = 0.5

    def __call__(self, results):
        if np.random.rand() >= self.flip_ratio:
            return results
        results = self.flip_bbox(results)
        results = self.flip_cam_params(results)
        results = self.flip_img(results)
        return results

    def flip_img(self, results, direction='horizontal'):
        results['img'] = [mmcv.imflip(img, direction) for img in results['img']]
        return results

    def flip_cam_params(self, results):
        flip_factor = np.eye(4)
        flip_factor[1, 1] = -1
        lidar2cam = [l2c @ flip_factor for l2c in results['lidar2cam']]
        w = results['img_shape'][0][1]
        lidar2img = []
        for cam_intrinsic, l2c in zip(results['cam_intrinsic'], lidar2cam):
            cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
            lidar2img.append(cam_intrinsic @ l2c)
        results['lidar2cam'] = lidar2cam
        results['lidar2img'] = lidar2img
        return results

    def flip_bbox(self, input_dict, direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
        return input_dict


@PIPELINES.register_module()
class ResizeMultiViewImage:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""

        results['img_shape'] = []
        results['pad_shape'] = []
        for i, input_img in enumerate(results['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    input_img,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = input_img.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    input_img,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results['img'][i] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'].append(img.shape)
            results['pad_shape'].append(img.shape)
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
            '''
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
            '''
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str



def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


def pixel_wise_transform(pixel_label, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        pixel_label (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    pixel_label[:, :2] = pixel_label[:, :2] * resize
    pixel_label[:, 0] -= crop[0]
    pixel_label[:, 1] -= crop[1]
    if flip:
        pixel_label[:, 0] = resize_dims[1] - pixel_label[:, 0]

    pixel_label[:, 0] -= W / 2.0
    pixel_label[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    pixel_label[:, :2] = np.matmul(rot_matrix, pixel_label[:, :2].T).T

    pixel_label[:, 0] += W / 2.0
    pixel_label[:, 1] += H / 2.0

    coords = pixel_label[:, :2].astype(np.int16)

    channels = pixel_label.shape[1] - 2

    pixel_map = np.zeros((*resize_dims, channels))
    valid_mask = ((coords[:, 1] < resize_dims[0])
                  & (coords[:, 0] < resize_dims[1])
                  & (coords[:, 1] >= 0)
                  & (coords[:, 0] >= 0))
    pixel_map[coords[valid_mask, 1],
              coords[valid_mask, 0]] = pixel_label[valid_mask, 2:]

    return torch.Tensor(pixel_map)



@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results["img"]
        N = len(imgs)
        new_imgs = []

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['cam_intrinsic'][i][:3, :3] = ida_mat @ results['cam_intrinsic'][i][:3, :3]

        results["img"] = new_imgs
        results['lidar2img'] = [results['cam_intrinsic'][i] @ results['lidar2cam'][i] for i in range(len(results['lidar2cam']))]


        results["vis_img"] = copy.deepcopy(new_imgs)


        if 'cam_gt_depth' in results:
            new_cam_gt_depth = []
            for i in range(len(results['cam_gt_depth'])):
                cam_gt_depth = results['cam_gt_depth'][i]
                cam_gt_depth_augmented = depth_transform(
                        cam_gt_depth, resize, self.data_aug_conf["final_dim"],
                        crop, flip, rotate)
                new_cam_gt_depth.append(cam_gt_depth_augmented)
            results["cam_gt_depth"] = new_cam_gt_depth

        if 'pixel_wise_label' in results:
            new_cam_gt_depth, new_pw_label = [], []
            for i in range(len(results['pixel_wise_label'])):
                pw_label = results['pixel_wise_label'][i]
                cam_gt_depth_augmented, pw_label_augmented = pixel_wise_transform(
                        pw_label, resize, self.data_aug_conf["final_dim"],
                        crop, flip, rotate)
                new_cam_gt_depth.append(cam_gt_depth_augmented)
                new_pw_label.append(pw_label_augmented)
            results["cam_gt_depth"] = new_cam_gt_depth
            results["pixel_wise_label"] = new_pw_label

        return results

    def _get_rot(self, h):

        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class CustomResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results["img"]
        N = len(imgs)
        new_imgs = []

        if 'img_trans_dict' in results:
            resize = results['img_trans_dict']['resize']
            resize_dims = results['img_trans_dict']['resize_dims']
            crop = results['img_trans_dict']['crop']
            flip = results['img_trans_dict']['flip']
            rotate = results['img_trans_dict']['rotate']
        else:
            resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
            results['img_trans_dict'] = dict(
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['cam_intrinsic'][i][:3, :3] = ida_mat @ results['cam_intrinsic'][i][:3, :3]

        results["img"] = new_imgs
        results['lidar2img'] = [results['cam_intrinsic'][i] @ results['lidar2cam'][i] for i in range(len(results['lidar2cam']))]

        if 'pixel_wise_label' in results:
            new_pw_label = []
            for i in range(len(results['pixel_wise_label'])):
                pw_label = results['pixel_wise_label'][i]
                pw_label_augmented = pixel_wise_transform(
                        pw_label, resize, self.data_aug_conf["final_dim"],
                        crop, flip, rotate)
                new_pw_label.append(pw_label_augmented)
            results["pixel_wise_label"] = new_pw_label

        if 'img_semantic' in results:
            new_img_semantic = []
            for i in range(N):
                img_semantic = Image.fromarray(np.uint8(results['img_semantic'][i]))
                # augmentation (resize, crop, horizontal flip, rotate)
                img_semantic, _ = self._img_transform(
                    img_semantic,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                    nearst=True,
                )
                new_img_semantic.append(np.array(img_semantic).astype(np.float32))
            results["img_semantic"] = new_img_semantic
            if 'pixel_wise_label' in results:
                for i in range(N):
                    results["pixel_wise_label"][i][..., 1] = results["pixel_wise_label"][i].new(new_img_semantic[i])

        return results

    def _get_rot(self, h):

        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate, nearst=None):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        if nearst is None:
            img = img.resize(resize_dims)
        else:
            img = img.resize(resize_dims, Image.NEAREST)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate

        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle *= -1

        if 'points' in results:
            points = results['points'].tensor
            points, rot_mat_T = results["gt_bboxes_3d"].rotate(
                np.array(rot_angle), points
            )
            results['points'].tensor = points
        else:
            results["gt_bboxes_3d"].rotate(
                np.array(rot_angle)
            )

        # rotate pixel_wise velo
        if 'pixel_wise_label' in results:
            new_pw_label = []
            rot_sin = np.sin(rot_angle)
            rot_cos = np.cos(rot_angle)
            rot_mat_T = np.array([[rot_cos, -rot_sin, 0],
                                 [rot_sin, rot_cos, 0],
                                 [0, 0, 1]])
            for i in range(len(results['pixel_wise_label'])):
                pw_label = results['pixel_wise_label'][i]
                if pw_label.shape[-1] == 7:
                    pw_label[:, 5:7] = np.dot(pw_label[:, 5:7], rot_mat_T[:2, :2])
                new_pw_label.append(pw_label)
            results["pixel_wise_label"] = new_pw_label


        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)

        results["gt_bboxes_3d"].scale(scale_ratio)
        if 'points' in results:
            points = results['points']
            points.scale(scale_ratio)
            results['points'] = points
            #results['pcd_scale_factor'] = scale_ratio

        # scale pixel_wise velo
        if 'pixel_wise_label' in results:
            new_pw_label = []
            for i in range(len(results['pixel_wise_label'])):
                pw_label = results['pixel_wise_label'][i]
                if pw_label.shape[-1] == 7:
                    pw_label[:, 5:7] = pw_label[:, 5:7] * scale_ratio
                new_pw_label.append(pw_label)
            results["pixel_wise_label"] = new_pw_label

        # TODO: support translation

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            # results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()

            results["lidar2cam"][view] = (torch.tensor(results["lidar2cam"][view]).float() @ rot_mat_inv).numpy()
        results["lidar2ego"] = (torch.tensor(results["lidar2ego"]).float() @ rot_mat_inv).numpy()

        return

    def scale_xyz(self, results, scale_ratio):
        rot_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            # results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()

            results["lidar2cam"][view] = (torch.tensor(results["lidar2cam"][view]).float() @ rot_mat_inv).numpy()
        results["lidar2ego"] = (torch.tensor(results["lidar2ego"]).float() @ rot_mat_inv).numpy()

        return


@PIPELINES.register_module()
class CustomGlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[0.0, 0.0],
        scale_ratio_range=[1.0, 1.0],
        translation_std=[0.0, 0.0, 0.0],
        reverse_angle=False,
        pc_range=[-40, -40, -1.0, 40, 40, 5.4],
        flip_hv_ratio=[0.0, 0.0],
        space_size=[200, 200, 16],
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.flip_hv_ratio = flip_hv_ratio

        self.reverse_angle = reverse_angle
        self.pc_range = pc_range
        self.space_size = space_size

    def build_transformation(self, results, rot_angle=None, scale_ratio=None, trans_factor=None, flip_mask=None):
        # random rotate
        if rot_angle is None:
            rot_angle = np.random.uniform(*self.rot_range)
        R_mat = self.rotate_bev_along_z(results, rot_angle)

        # random scale
        if scale_ratio is None:
            scale_ratio = np.random.uniform(*self.scale_ratio_range)
        S_mat = self.scale_xyz(results, scale_ratio)

        # random trans
        if trans_factor is None:
            translation_std = np.array(self.translation_std, dtype=np.float32)
            trans_factor = np.random.normal(scale=translation_std, size=3).T.astype(np.float32)
        T_mat = self.translate_xyz(results, trans_factor)

        # flip
        if flip_mask is None:
            flip_horizontal = True if np.random.rand() < self.flip_hv_ratio[0] else False
            flip_vertical = True if np.random.rand() < self.flip_hv_ratio[1] else False
            flip_mask = [flip_horizontal, flip_vertical]
        F_mat = self.flip_xyz(results, flip_mask)

        transformation_matrix = (R_mat @ S_mat @ T_mat @ F_mat)
        results['ego_trans_dict']['matrix'] = transformation_matrix.numpy()
        results['ego_trans_dict']['rot_angle'] = rot_angle
        results['ego_trans_dict']['scale_ratio'] = scale_ratio
        results['ego_trans_dict']['trans_factor'] = trans_factor
        results['ego_trans_dict']['flip_mask'] = flip_mask

        # The ego coordinate is GT coordinate
        results["ego2lidar"] = (torch.tensor(results["ego2lidar"]).float() @ transformation_matrix).numpy()
        results["ego2global"] = (torch.tensor(results["ego2global"]).float() @ transformation_matrix).numpy()

        if "cam2ego" in results:
            new_cam2egos = []
            for cam2ego in results["cam2ego"]:
                ego2cam = np.linalg.inv(cam2ego)
                ego2cam = (torch.tensor(ego2cam).float() @ transformation_matrix).numpy()
                new_cam2egos.append(np.linalg.inv(ego2cam))
            results["cam2ego"] = new_cam2egos
        return

    def get_occ_reference_point(self):
        W, H, Z = self.space_size
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
        return occ_points.view(-1, 3)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate

        if 'occ_points' not in results:
            results['occ_points'] = self.get_occ_reference_point()
        if 'gt_bboxes_3d' not in results:
            #empty bbox_3d for transformation
            results['gt_bboxes_3d'] = results['box_type_3d'](
                np.array([], dtype=np.float32))

        rot_angle, scale_ratio, trans_factor, flip_mask = None, None, None, None
        if 'ego_trans_dict' in results:
            rot_angle = results['ego_trans_dict']['rot_angle']
            scale_ratio = results['ego_trans_dict']['scale_ratio']
            trans_factor = results['ego_trans_dict']['trans_factor']
            flip_mask = results['ego_trans_dict']['flip_mask']

        results['ego_trans_dict'] = dict()
        self.build_transformation(results, rot_angle, scale_ratio, trans_factor, flip_mask)

        # remap GT_occ by using reference points
        if 'voxel_semantics' in results:
            new_semantics = np.zeros_like(results['voxel_semantics'])
            new_mask_lidar = np.zeros_like(results['mask_lidar'])
            new_mask_camera = np.zeros_like(results['mask_camera'])

            W, H, Z = results['voxel_semantics'].shape
            grid_size = [(self.pc_range[3] - self.pc_range[0]) / W,
                        (self.pc_range[4] - self.pc_range[1]) / H,
                        (self.pc_range[5] - self.pc_range[2]) / Z]

            new_coords = results['occ_points'].clone()
            new_coords[..., 0] = (new_coords[..., 0] - self.pc_range[0]) / grid_size[0]
            new_coords[..., 1] = (new_coords[..., 1] - self.pc_range[1]) / grid_size[1]
            new_coords[..., 2] = (new_coords[..., 2] - self.pc_range[2]) / grid_size[2]
            new_coords = new_coords.int()
            mask = (new_coords[..., 0] >= 0) &(new_coords[..., 0] < W) & \
                (new_coords[..., 1] >= 0) &(new_coords[..., 1] < H) & \
                (new_coords[..., 2] >= 0) &(new_coords[..., 2] < Z)
            new_coords = new_coords[mask]
            new_semantics[new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]] = results['voxel_semantics'].reshape(-1)[mask]
            new_mask_lidar[new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]] = results['mask_lidar'].reshape(-1)[mask]
            new_mask_camera[new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]] = results['mask_camera'].reshape(-1)[mask]

            results['voxel_semantics'] = new_semantics
            results['mask_lidar'] = new_mask_lidar
            results['mask_camera'] = new_mask_camera

            if 'voxel_vel' in results:
                new_voxel_vel = np.zeros_like(results['voxel_vel'])
                new_voxel_vel[new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]] = results['voxel_vel'].reshape(-1, 2)[mask]
                voxe_vel = torch.from_numpy(new_voxel_vel)
                voxe_vel = torch.cat([voxe_vel, torch.zeros_like(voxe_vel[..., 0:1])], dim=-1)
                ego_trans_R = results['ego_trans_dict']['matrix'][:3, :3]

                voxe_vel = (torch.from_numpy(np.linalg.inv(ego_trans_R))[None, None, None, :, :] @ voxe_vel.unsqueeze(-1)).squeeze(-1)
                voxe_vel = voxe_vel[..., :2]
                results['voxel_vel'] = voxe_vel

        return results

    def rotate_bev_along_z(self, results, angle):

        # TODO: the default gt_bboxes_3d is in lidar coords, if you want to make aug for it, you should trans it into ego coords first
        #rot_angle = -1 * angle if self.reverse_angle else angle
        #results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        # unclock-wise
        rot_mat = torch.tensor([
            [rot_cos, -rot_sin, 0, 0], 
            [rot_sin, rot_cos, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]])

        occ_points = results['occ_points']
        occ_points = torch.cat([occ_points, torch.ones_like(occ_points[:, 0:1])], dim=-1)
        occ_points = (rot_mat[None, :, :] @ occ_points.unsqueeze(-1)).squeeze(-1)[:, :3]
        results['occ_points'] = occ_points

        rot_mat_inv = torch.inverse(rot_mat)

        return rot_mat_inv

    def scale_xyz(self, results, scale_ratio):
        
        #results["gt_bboxes_3d"].scale(scale_ratio)
        occ_points = results['occ_points']
        occ_points *= scale_ratio
        results['occ_points'] = occ_points
        
        rot_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        rot_mat_inv = torch.inverse(rot_mat)

        return rot_mat_inv

    def translate_xyz(self, results, trans_factor):

        #results["gt_bboxes_3d"].translate(trans_factor)
        occ_points = results['occ_points']
        occ_points += trans_factor
        results['occ_points'] = occ_points

        rot_mat = torch.tensor(
            [
                [1.0, 0, 0, trans_factor[0]],
                [0, 1.0, 0, trans_factor[1]],
                [0, 0, 1.0, trans_factor[2]],
                [0, 0, 0, 1],
            ]
        )

        rot_mat_inv = torch.inverse(rot_mat)

        return rot_mat_inv

    def flip_xyz(self, results, flip_mask):

        flip_horizontal, flip_vertical = flip_mask
        occ_points = results['occ_points']
        if flip_horizontal:
            #results["gt_bboxes_3d"].flip('horizontal')
            occ_points[..., 1] = -occ_points[..., 1]
        if flip_vertical:
            #results["gt_bboxes_3d"].flip('vertical')
            occ_points[..., 0] = -occ_points[..., 0]

        results['occ_points'] = occ_points

        rot_x = -1.0 if flip_vertical else 1.0
        rot_y = -1.0 if flip_horizontal else 1.0
        flip_mat = torch.tensor(
            [
                [rot_x, 0, 0, 0],
                [0, rot_y, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        flip_mat_inv = torch.inverse(flip_mat)

        return flip_mat_inv