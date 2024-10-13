# Train & Evaluation

## Train
Multi-GPU (8 GPUs):
```bash
./tools/torchrun_occ_train.sh ./projects/configs/viewformer/viewformer_r50_704x256_seq_90e.py 8 --work-dir ./work_dirs/your_folder_name
```

Single-GPU:
```bash
./occ_train.sh
```



## Evaluation
Multi-GPU (8 GPUs):
```bash
./tools/torchrun_occ_test.sh ./projects/configs/viewformer/viewformer_r50_704x256_seq_90e.py ./ckpts/viewformer_res50_704x256_depthpretrain_90e.pth 8 --eval occ
```

Single-GPU:
```bash
./occ_test.sh
```

The evaluation results should be as follow:
```
occ_eval - INFO - ===> per class IoU of 6019 samples:
occ_eval - INFO - ===> others - IoU = 12.94
occ_eval - INFO - ===> barrier - IoU = 50.11
occ_eval - INFO - ===> bicycle - IoU = 27.97
occ_eval - INFO - ===> bus - IoU = 44.61
occ_eval - INFO - ===> car - IoU = 52.85
occ_eval - INFO - ===> construction_vehicle - IoU = 22.38
occ_eval - INFO - ===> motorcycle - IoU = 29.62
occ_eval - INFO - ===> pedestrian - IoU = 28.01
occ_eval - INFO - ===> traffic_cone - IoU = 29.28
occ_eval - INFO - ===> trailer - IoU = 35.18
occ_eval - INFO - ===> truck - IoU = 39.4
occ_eval - INFO - ===> driveable_surface - IoU = 84.71
occ_eval - INFO - ===> other_flat - IoU = 49.39
occ_eval - INFO - ===> sidewalk - IoU = 57.44
occ_eval - INFO - ===> terrain - IoU = 59.69
occ_eval - INFO - ===> manmade - IoU = 47.37
occ_eval - INFO - ===> vegetation - IoU = 40.56
occ_eval - INFO - ===> free - IoU = 90.06
occ_eval - INFO - ===> mIoU of 6019 samples: 41.85
occ_eval - INFO - ===> geo - IoU: 71.25
```