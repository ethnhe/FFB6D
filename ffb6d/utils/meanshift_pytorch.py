#!/usr/bin/env python3
import os
import cv2
import time
import math
import torch
import numpy as np
import pickle as pkl

from sklearn.cluster import MeanShift
from sklearn.neighbors import KDTree
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


def gaussian_kernel(distance, bandwidth):
    return torch.exp(-0.5 * ((distance / bandwidth)) ** 2) \
        / (bandwidth * math.sqrt(2 * np.pi))


def distance_batch(a, b):
    return torch.sqrt(((a[None, :] - b[:, None]) ** 2).sum(2))


class MeanShiftTorch():
    def __init__(self, bandwidth=0.05, max_iter=300):
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter

    def fit(self, A, ret_mid_res=False):
        # params: A: [N, 3]
        N, c = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            dis = torch.norm(C.reshape(1, N, c) - C.reshape(N, 1, c), dim=2)
            w = gaussian_kernel(dis, self.bandwidth).reshape(N, N, 1)
            new_C = torch.sum(w * C, dim=1) / torch.sum(w, dim=1)
            # new_C = C + shift_offset
            Cdis = torch.norm(new_C - C, dim=1)
            # print(C, new_C)
            C = new_C
            if torch.max(Cdis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        dis = torch.norm(C.view(N, 1, c) - C.view(1, N, c), dim=2)
        num_in = torch.sum(dis < self.bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        labels = dis[max_idx] < self.bandwidth
        if not ret_mid_res:
            return C[max_idx, :], labels
        else:
            return C, dis

    def fit_multi_clus(self, A):
        # params: A: [N, 3]
        C, dis = self.fit(A, ret_mid_res=True)

        num_in = torch.sum(dis < self.bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        iclus = 1
        labels = (dis[max_idx] < self.bandwidth).int() * iclus

        C_lst = [C[max_idx, :]]
        n_in_lst = [max_num.item()]
        while True:
            iclus += 1
            if (labels == 0).sum() < 1:
                break
            C_rm = C[labels == 0, :]
            dis = torch.norm(C_rm.unsqueeze(0) - C_rm.unsqueeze(1), dim=2)
            num_in = torch.sum(dis < self.bandwidth, dim=1)
            max_num, max_idx = torch.max(num_in, 0)
            lb_idxs = torch.arange(labels.shape[0])
            in_lb_idxs = lb_idxs[labels == 0][dis[max_idx] < self.bandwidth]
            labels[in_lb_idxs] = iclus
            C_lst.append(C_rm[max_idx, :])
            n_in_lst.append(max_num.item())

        return C_lst, labels, n_in_lst


class MeanShiftTorchWithFor():
    def __init__(self, bandwidth=0.05, max_iter=300):
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter

    def distance(self, a, A):
        return torch.sqrt(((a - A)**2).sum(1))

    def gaussian(self, dist):
        return torch.exp(-.5 * ((dist / self.bandwidth))**2) / (self.bandwidth * math.sqrt(2 * math.pi))

    def meanshift_step(self, A):
        for i, a in enumerate(A):
            dist = self.distance(a, A)
            weight = self.gaussian(dist)
            A[i] = (weight[:, None] * A).sum(0) / weight.sum()
        return A

    def fit(self, A):
        # params: A: [N, 3]
        for it in range(1):
            A = self.meanshift_step(A)
        return A

    def fit_batch(self, A, batch_size=2500):
        n = A.shape[0]
        for _ in range(5):
            for i in range(0, n, batch_size):
                s = slice(i, min(n, i + batch_size))
                print(s, A.shape)
                weight = self.gaussian(distance_batch(A, A[s]))
                print(weight.shape, A.shape)
                from IPython import embed
                embed()
                num = (weight[:, :, None] * A).sum(dim=1)
                A[s] = num / weight.sum(1)[:, None]
        return A


def test():
    while True:
        # a = np.random.rand(20000, 2)
        n_clus = 5
        n_samples = 100
        bw = 10
        centroids = np.random.uniform(0, 480, (n_clus, 2))
        slices = [np.random.multivariate_normal(centroids[i], np.diag([50., 50.]), n_samples+i*100)
                  for i in range(n_clus)]
        a = np.concatenate(slices).astype(np.float32)
        print("npts:", a.shape)
        ta = torch.from_numpy(a.astype(np.float32)).cuda()

        a_idx = (a / a.max() * 480).astype("uint8")
        show_a = np.zeros((480, 480, 3), dtype="uint8")
        show_a[a_idx[:, 0], a_idx[:, 1], :] = np.array([255, 255, 255])

        ms = MeanShiftTorch(bw)
        ctr, lb = ms.fit(ta)
        ctr = (ctr.cpu().numpy() / a.max() * 480).astype("uint8")
        show_a_one = cv2.circle(show_a.copy(), (ctr[1], ctr[0]), 3, (0, 0, 255), -1)

        ctr_lst, lb, n_in_lst = ms.fit_multi_clus(ta)
        print(ctr_lst, n_in_lst)
        show_a_multi = show_a.copy()
        for ctr in ctr_lst:
            ctr = (ctr.cpu().numpy() / a.max() * 480).astype("uint8")
            show_a_multi = cv2.circle(show_a_multi, (ctr[1], ctr[0]), 3, (0, 0, 255), -1)

        def get_color(cls_id, n_obj=30):
            mul_col = 255 * 255 * 255 // n_obj * cls_id
            r, g, b = mul_col // 255 // 255, (mul_col // 255) % 255, mul_col % 255
            bgr = (int(r), int(g), int(b))
            return bgr

        show_ca = np.zeros((480, 480, 3), dtype="uint8")
        print(lb.unique())
        n_clus = lb.max()
        for cls in range(1, n_clus+1):
            inl = a_idx[lb.cpu().numpy() == cls, :]
            show_ca[inl[:, 0], inl[:, 1], :] = np.array(
                list(get_color(cls, n_obj=n_clus+1))
            )

        # ms_cpu = MeanShift(
        #     bandwidth=bw, n_jobs=8
        # )
        # ms_cpu.fit(a)
        # clus_ctrs = np.array(ms_cpu.cluster_centers_)
        # clus_labels = ms_cpu.labels_
        # ctr = (clus_ctrs[0] / a.max() * 480).astype("uint8")
        # show_a = cv2.circle(show_a, (ctr[1], ctr[0]), 3, (255, 0, 0), -1)
        # imshow("show_b", show_b)

        imshow('show_a_one', show_a_one)
        imshow("show_a_multi", show_a_multi)
        imshow('show_ca', show_ca)
        waitKey(0)


def main():
    test()


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
