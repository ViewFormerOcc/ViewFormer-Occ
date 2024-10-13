import os
import tqdm
import json
from visual_nuscenes import NuScenes
use_gt = False
#out_dir = './work_dirs/20221129_convnextxl_addDepth_1600640_sweeps4fu4_inter1_trainval/result_vis'
#result_json = "./work_dirs/20221129_convnextxl_addDepth_1600640_sweeps4fu4_inter1_trainval/results_test/pts_bbox/results_nusc"
#out_dir = './work_dirs/20220980_vovnet_add2Depthloss_800320_sweep5f30/result_vis'
#result_json = "./work_dirs/20220980_vovnet_add2Depthloss_800320_sweep5f30/results_val/pts_bbox/results_nusc"
out_dir = './work_dirs/20221125_vovnet_addDepth_704256_2depthloss_stepLR_NobackboneFreeze_DN_sweepskey2fu2_timeInt1/result_vis'
result_json = "./work_dirs/20221125_vovnet_addDepth_704256_2depthloss_stepLR_NobackboneFreeze_DN_sweepskey2fu2_timeInt1/results_val/pts_bbox/results_nusc"
dataroot='./data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)
    #nusc = NuScenes(version='v1.0-test', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())

# 1320 -- 1500
for token in tqdm.tqdm(tokens[1405:1410]):
    if use_gt:
        nusc.render_sample(token, out_path = out_dir+"/"+token+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path = out_dir+"/"+token+"_pred.png", verbose=False)

