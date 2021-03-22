#!/usr/bin/env python3
import os
from helper_tool import DataProcessing as DP
import numpy as np
import pickle as pkl
from RandLANet import Network
from tqdm import tqdm
from boxx import *

import torch
import torch.nn as nn


class ConfigTest:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 480 * 640 # Number of input points
    num_classes = 22  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 1  # batch_size during training
    val_batch_size = 1  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch
    in_c = 6

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [8, 32, 64, 128]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


def main():
    cfg = ConfigTest
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Network(cfg).to(device)
    print("model parameters:", sum(param.numel() for param in net.parameters()))

    for i in tqdm(range(10)):
        npts = cfg.num_points
        pcld = np.random.rand(1, npts, 3)
        feat = np.random.rand(1, 6, npts)
        n_layers = 4
        sub_s_r = [16, 1, 4, 1]
        inputs = {}
        for i in range(n_layers):
            nei_idx = DP.knn_search(pcld, pcld, 16)
            sub_pts = pcld[:, :pcld.shape[1] // sub_s_r[i], :]
            pool_i = nei_idx[:, :pcld.shape[1] // sub_s_r[i], :]
            up_i = torch.LongTensor(DP.knn_search(sub_pts, pcld, 1))
            inputs['xyz'] = inputs.get('xyz', []) + [torch.from_numpy(pcld).float().to(device)]
            inputs['neigh_idx'] = inputs.get('neigh_idx', []) + [torch.LongTensor(nei_idx).to(device)]
            inputs['sub_idx'] = inputs.get('sub_idx', []) + [torch.LongTensor(pool_i).to(device)]
            inputs['interp_idx'] = inputs.get('interp_idx', []) + [torch.LongTensor(up_i).to(device)]
            pcld = sub_pts
        inputs['features'] = torch.from_numpy(feat).float().to(device)

        end_points = net(inputs)

    for k, v in end_points.items():
        if type(v) == list:
            for ii, item in enumerate(v):
                print(k+'%d'%ii, item.size())
        else:
            print(k, v.size())


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
