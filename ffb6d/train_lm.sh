#!/bin/bash
n_gpu=8
cls='ape'
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls
