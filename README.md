<div align="center">
<h2>ViewFormer: Exploring Spatiotemporal Modeling for Multi-View 3D Occupancy Perception via View-Guided Transformers</h2>
</div>

<video src="https://github.com/user-attachments/assets/a5856329-0210-4e3a-bbfb-16580e47ba9e" controls="controls" width="500" height="300"></video>

> **ViewFormer: Exploring Spatiotemporal Modeling for Multi-View 3D Occupancy Perception via View-Guided Transformers**, ECCV 2024
> - [Paper in arXiv](https://arxiv.org/abs/2405.04299) | [Blog in Chinese](https://zhuanlan.zhihu.com/p/706548179)

# News
- [2024/7/01]: 🚀 ViewFormer is accepted by **ECCV 2024**.
- [2024/5/15]: 🚀 ViewFormer ranks **1st** on the occupancy trick of [RoboDrive Challenge](https://robodrive-24.github.io/)!


# Abstract

3D occupancy, an advanced perception technology for driving scenarios, represents the entire scene without distinguishing between foreground and background by quantifying the physical space into a grid map. The widely adopted projection-first deformable attention, efficient in transforming image features into 3D representations, encounters challenges in aggregating multi-view features due to sensor deployment constraints. To address this issue, we propose our learning-first view attention mechanism for effective multi-view feature aggregation. Moreover, we showcase the scalability of our view attention across diverse multi-view 3D tasks, including map construction and 3D object detection. Leveraging the proposed view attention as well as an additional multi-frame streaming temporal attention, we introduce ViewFormer, a vision-centric transformer-based framework for spatiotemporal feature aggregation. To further explore occupancy-level flow representation, we present FlowOcc3D, a benchmark built on top of existing high-quality datasets. Qualitative and quantitative analyses on this benchmark reveal the potential to represent fine-grained dynamic scenes. Extensive experiments show that our approach significantly outperforms prior state-of-the-art methods.

# Methods

<div align="center">
  <img src="figs/framework.png" width="800"/>
</div><br/>

<div align="center">
  <img src="figs/task.png" width="800"/>
</div><br/>

## Getting Started

Please follow our documentations to get started.

1. [**Environment Setup.**](./docs/setup.md)
2. [**Data Preparation.**](./docs/data_preparation.md)
3. [**Training and Inference.**](./docs/training_inference.md)



## Results on [Occ3D](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction/tree/main)(based on nuScenes) Val Set.
| Method | Backbone | Pretrain | Lr Schd | mIoU |  Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViewFormer | R50 | [R50-depth](https://github.com/zhiqi-li/storage/releases/download/v1.0/r50_256x705_depth_pretrain.pth) | 90ep | 41.85 |[config](projects/configs/viewformer/viewformer_r50_704x256_seq_90e.py) |[model](https://drive.google.com/file/d/1_8ZD0IvtO7_T5l4TxRflQ3q07Oa7pcKB/view?usp=sharing)|
| ViewFormer | InternT | [COCO](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_3x_coco.pth) | 24ep | 43.61 |[config](projects/configs/viewformer/viewformer_InternImageTiny_704x256_seq.py) |[model](https://drive.google.com/file/d/1meFK7NEJml6yLmQeBrqdr12NnQ_jC9Ya/view?usp=sharing)|

**Note**: 
- Since we do not adopt the CBGS setting, our 90-epoch schedule is equivalent to the 20-epoch schedule in FB-OCC, which extends the training period by approximately 4.5 times.

## Results on [FlowOcc3D](https://huggingface.co/viewformer/ViewFormer-Occ) Val Set.
| Method | Backbone | Pretrain | Lr Schd | mIoU | mAVE |  Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViewFormer | InternT | [COCO](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_3x_coco.pth) | 24ep | 42.54 | 0.412 |[config](projects/configs/viewformer/viewformer_InternImageTiny_704x256_seq_flow.py) |[model](https://drive.google.com/file/d/1rkGHPtmryjLIZOEk-4asaDegErGOX9Uy/view?usp=sharing)|

**Note**: 
- The difference between COCO pre-trained weights and ImageNet pre-trained weights in our experiments is minimal. ImageNet pre-trained weights achieve slightly higher accuracy. We maintain the COCO pre-trained weights here to fully replicate the accuracy reported in our paper.

## Acknowledgements

We are grateful for these great works as well as open source codebases.

* 3D Occupancy: [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D),
[OccNet](https://github.com/OpenDriveLab/OccNet),
[FB-OCC](https://github.com/NVlabs/FB-BEV).
* 3D Detection: [MMDetection3d](https://github.com/open-mmlab/mmdetection3d), [DETR3D](https://github.com/WangYueFt/detr3d), [PETR](https://github.com/megvii-research/PETR), [BEVFormer](https://github.com/fundamentalvision/BEVFormer),
[BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth),
[SOLOFusion](https://github.com/Divadi/SOLOFusion), [StreamPETR](https://github.com/exiawsh/StreamPETR).


Please also follow our visualization tool [Oviz](https://github.com/xiaoqiang-cheng/Oviz), if you are interested in the visualization in our paper.


## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
    @article{li2024viewformer,
        title={ViewFormer: Exploring Spatiotemporal Modeling for Multi-View 3D Occupancy Perception via View-Guided Transformers}, 
        author={Jinke Li and Xiao He and Chonghua Zhou and Xiaoqiang Cheng and Yang Wen and Dan Zhang},
        journal={arXiv preprint arXiv:2405.04299},
        year={2024},
    }
```
