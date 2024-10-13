# Data Preparation

## 1. Dataset

Please follow instructions of [CVPR2023 3D Occupancy Challenge](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) to download nuScenes dataset, which need to be placed in `./data/nuscenes`.

## 2. [FlowOcc3D](https://huggingface.co/viewformer/ViewFormer-Occ) (3D Occupancy Flow Dataset)

Download [flow GT](https://huggingface.co/viewformer/ViewFormer-Occ). Unzip it in `./data/nuscenes`.

## 3. Creating infos file

```python
python tools/create_occ_data.py occ --root-path ./data/occ3d-nus --out-dir ./data/nuscenes --extra-tag occ_infos_temporal --version v1.0-trainval --canbus ./data/nuscenes --occ-path ./data/nuscenes/occ3d-nus
```

Generate `occ_infos_temporal_{train,val}.pkl` via this command.

Additionally, we also provide an exisitng [train](https://drive.google.com/file/d/1GO073_4bwRDyE9ObsNqSozJLe6PCuxcn/view?usp=sharing) / [val](https://drive.google.com/file/d/1WzlUkmGHKOQzVexJ1NN4vukCBiDoAfnu/view?usp=sharing) pkl for a quick start.

## 4. Pretrained Weights
Download pre-trained weights on our homepage and put them in the ``./ckpts`` folder.
```shell
cd /path/to/ViewFormer-Occ
mkdir ckpts
```

## 5. Auxiliary Depth Supervision (Training Only)
For this part of the code, we followed the early implementation of BEVDepth, where we first save the ground truth depth images for each frame and then load them during training. Download the BEVDepth [train](https://drive.google.com/file/d/1gfgdRpMWbESVX7OHkIIHj4ivF4Uljlnv/view?usp=sharing) / [val](https://drive.google.com/file/d/12HPYrdlpnEs2sP9gL-pgQdT_xO_2fMrC/view?usp=sharing) pkl, put them in ``./data/nuscenes``. Then run the following script.
```shell
python scripts/gen_depth_label_gt.py
```
Additionally, we also provide generated [depth GT files](https://drive.google.com/file/d/1AVpSpXy-RQMwCfozvvu5DbmnsO7Znqu0/view?usp=sharing) for a quick start. Unzip it in ``./data/nuscenes``.

If you don't want to use auxiliary depth supervision, you don't have to generate or download this part of the data and set ``depth_supvise=False`` in the config.


**The final directory structure should be as follow:**
```
ViewFormer-Occ
├── projects/
├── mmdetection3d/
├── tools/
├── ckpts/
├── data/
│   ├── nuscenes/
│   │   ├── occ_flow_sparse_ext/
│   │   ├── depth_pano_seg_gt/
│   │   ├── can_bus/
│   │   ├── gts/
│   │   ├── maps/
│   │   ├── occ3d-nus/
│   │   ├── samples/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── occ_infos_temporal_train.pkl
|   |   ├── occ_infos_temporal_val.pkl
```
