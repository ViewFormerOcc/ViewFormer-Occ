PYTHONPATH="$(dirname $0)/..":$PYTHONPATHi \
CUDA_VISIBLE_DEVICES=0 python tools/occ_test.py ./projects/configs/viewformer/viewformer_r50_704x256_seq_90e.py ./ckpts/viewformer_res50_704x256_depthpretrain_90e.pth --eval occ
