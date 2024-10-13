PYTHONPATH="$(dirname $0)/..":$PYTHONPATHi \
CUDA_VISIBLE_DEVICES=0 python tools/occ_train.py ./projects/configs/viewformer/viewformer_r50_704x256_seq_90e.py --work-dir ./work_dirs/your_folder_name
