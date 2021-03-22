#!/usr/bin/env python3
import os
import cv2
import time
import torch
import numpy as np
import pickle as pkl

from sklearn.cluster import MeanShift
from sklearn.neighbors import KDTree
from cv2 import imshow, waitKey


def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * torch.sqrt(2 * torch.tensor(np.pi)))) \
        * torch.exp(-0.5 * ((distance / bandwidth)) ** 2)


def distance_batch(a, b):
    return torch.sqrt(((a[None, :] - b[:, None]) ** 2).sum(2))


class BatchMeanShiftTorch():
    def __init__(
        self, bandwidth=0.05, max_iter=300, batch_npts=5000, max_cnt=1e5
    ):
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter
        self.ms_bs = batch_npts
        self.max_cnt = max_cnt

    def gaussian(self, d, bw):
        return torch.exp(-0.5*((d/bw))**2) / (bw*torch.sqrt(2*torch.tensor(np.pi)))

    def dist(self, a, b):
        """
        Params:
            a: [N1, c]
            b: [N2, c]
        Return:
            [N2, N1, 1] of distance
        """
        return torch.sqrt((a.unsqueeze(0) - b.unsqueeze(1))**2).sum(2)

    def dist_bs(self, a, b):
        """
        Params:
            a: [bs, N1, c]
            b: [bs, N2, c]
        Return:
            [bs, N2, N1, 1] of distance
        """
        return torch.sqrt((a.unsqueeze(1) - b.unsqueeze(2))**2).sum(3)

    def sum_sqz(self, a, axis):
        return a.sum(axis).squeeze(axis)

    def fit(self, A):
        """
        Params:
            A: [N, 3]
        """
        N, c = A.size()
        if N > self.max_cnt:
            c_mask = np.zeros(N, dtype=int)
            c_mask[:self.max_cnt] = 1
            np.random.shuffle(c_mask)
            A = A[c_mask, :]
        it = 0
        while True:
            it += 1
            max_dis = 0.0
            for i in range(0, N, self.ms_bs):
                s = slice(i, min(N, i+self.ms_bs))
                dis = self.dist(A, A[s])
                w = self.gaussian(dis, self.bandwidth).unsqueeze(-1)
                num = self.sum_sqz(torch.mul(w, A), 1)
                oA = A[s].clone()
                A[s] = num / self.sum_sqz(w, 1).unsqueeze(1)
                dif_dis = torch.norm(A[s] - oA, dim=1)
                t_max = torch.max(dif_dis).item()
                if t_max > max_dis:
                    max_dis = t_max
            if max_dis < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break

        # find biggest cluster
        max_num, max_idx = 0, 0
        for i in range(0, N, self.ms_bs):
            s = slice(i, min(N, i+self.ms_bs))
            dis = self.dist(A, A[s])
            num_in = torch.sum(dis < self.bandwidth, dim=1)
            t_max_num, t_max_idx = torch.max(num_in, 0)
            if t_max_num > max_num:
                max_num = t_max_num
                max_idx = t_max_idx + i

        dis = torch.norm(A - A[max_idx, :].unsqueeze(0), dim=1)
        labels = dis < self.bandwidth
        return A[max_idx, :], labels

    def fit_batch(self, A):
        bs, N, c = A.size()
        it = 0
        while True:
            it += 1
            max_dis = 0.0
            for i in range(0, N, self.ms_bs):
                s = slice(i, min(N, i+self.ms_bs))
                dis = self.dist_bs(A, A[:, s])
                w = self.gaussian(dis, self.bandwidth).unsqueeze(-1)
                num = self.sum_sqz(torch.mul(w, A.unsqueeze(1)), 2)
                oA = A[:, s].clone()
                A[:, s] = num / self.sum_sqz(w, 2).unsqueeze(2)
                dif_dis = torch.norm(A[:, s] - oA, dim=2)
                t_max = torch.max(dif_dis).item()
                if t_max > max_dis:
                    max_dis = t_max
            if max_dis < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break

        # find biggest cluster
        mn, mi = None, None
        for i in range(0, N, self.ms_bs):
            s = slice(i, min(N, i+self.ms_bs))
            dis = self.dist_bs(A, A[:, s])
            num_in = torch.sum(dis < self.bandwidth, dim=2)
            t_mn, t_mi = torch.max(num_in, 1)
            t_mi += i
            if i == 0:
                mn, mi = t_mn, t_mi
            else:
                t_mn = torch.cat((mn.unsqueeze(1), t_mn.unsqueeze(1)), dim=1)
                t_mi = torch.cat((mi.unsqueeze(1), t_mi.unsqueeze(1)), dim=1)
                mn, si = torch.max(t_mn, dim=1)
                mi = torch.gather(t_mi, 1, si.unsqueeze(1)).squeeze(1)

        A_max = torch.gather(A, 1, mi.view(bs, 1, 1).repeat(1, 1, c))
        dis = torch.norm(A - A_max, dim=2)
        labels = dis < self.bandwidth
        return A_max.squeeze(), labels


class MeanShiftTorch():
    def __init__(self, bandwidth=0.05, max_iter=300):
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter

    def fit(self, A):
        # params: A: [N, 3]
        N, c = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(1, N, c).repeat(N, 1, 1)
            Cr = C.view(N, 1, c).repeat(1, N, 1)
            dis = torch.norm(Cr - Ar, dim=2)
            w = gaussian_kernel(dis, self.bandwidth).view(N, N, 1)
            new_C = torch.sum(w * Ar, dim=1) / torch.sum(w, dim=1)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=1)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, c).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=2)
        num_in = torch.sum(dis < self.bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        labels = dis[max_idx] < self.bandwidth
        return C[max_idx, :], labels

    def fit_batch_npts(self, A):
        # params: A: [bs, n_kps, pts, 3]
        bs, n_kps, N, cn = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(bs, n_kps, 1, N, cn).repeat(1, 1, N, 1, 1)
            Cr = C.view(bs, n_kps, N, 1, cn).repeat(1, 1, 1, N, 1)
            dis = torch.norm(Cr - Ar, dim=4)
            w = gaussian_kernel(dis, self.bandwidth).view(bs, n_kps, N, N, 1)
            new_C = torch.sum(w * Ar, dim=3) / torch.sum(w, dim=3)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=3)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, c).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=4)
        num_in = torch.sum(dis < self.bandwidth, dim=3)
        max_num, max_idx = torch.max(num_in, 2)
        dis = torch.gather(dis, 2, max_idx.reshape(bs, n_kps, 1))
        labels = dis < self.bandwidth
        ctrs = torch.gather(
            C, 2, max_idx.reshape(bs, n_kps, 1, 1).repeat(1, 1, 1, cn)
        )
        return ctrs, labels
# """


def test():
    while True:
        # a = np.random.rand(20000, 2)
        n_clus = 5
        n_samples = 10000
        bw = 2
        centroids = np.random.uniform(0, 480, (n_clus, 2))
        slices = [np.random.multivariate_normal(centroids[i], np.diag([5., 5.]), n_samples+i*100)
                  for i in range(n_clus)]
        a = np.concatenate(slices).astype(np.float32)
        print("npts:", a.shape)
        ta = torch.from_numpy(a.astype(np.float32)).cuda()

        a_idx = (a / a.max() * 480).astype("uint8")
        show_a = np.zeros((480, 480, 3), dtype="uint8")
        show_a[a_idx[:, 0], a_idx[:, 1], :] = np.array([255, 255, 255])

        # ms = MeanShiftTorch(bw)
        # ctr, _ = ms.fit(ta)
        # ctr = (ctr.cpu().numpy() / a.max() * 480).astype("uint8")
        # print("pt_ctr:", ctr)
        # show_a = cv2.circle(show_a, (ctr[1], ctr[0]), 3, (0, 0, 255), -1)

        b_ms = BatchMeanShiftTorch(bw, batch_size=5000)
        ctr, labels = b_ms.fit(ta)
        for idx, lb in zip(a_idx, labels):
            if lb.item():
                show_a[idx[0], idx[1], :] = np.array([255, 255, 0])
        ctr = (ctr.cpu().numpy() / a.max() * 480).astype("uint8")
        show_a = cv2.circle(show_a, (ctr[1], ctr[0]), 3, (255, 0, 0), -1)
        print("bs_ctr:", ctr)

        # ms_cpu = MeanShift(
        #     bandwidth=bw, n_jobs=8
        # )
        # ms_cpu.fit(a)
        # clus_ctrs = np.array(ms_cpu.cluster_centers_)
        # clus_labels = ms_cpu.labels_
        # ctr = (clus_ctrs[0] / a.max() * 480).astype("uint8")
        # show_a = cv2.circle(show_a, (ctr[1], ctr[0]), 3, (255, 0, 0), -1)
        # imshow("show_b", show_b)
        imshow('show_a', show_a)
        waitKey(0)


def test_bs():
    while True:
        n_clus = 2
        n_samples = 5000
        bw = 2
        centroids = np.random.uniform(0, 480, (n_clus, 2))
        slices = [np.random.multivariate_normal(centroids[i], np.diag([5., 5.]), n_samples+i*100)
                  for i in range(n_clus)]
        a = np.concatenate(slices).astype(np.float32)
        # print("npts:", a.shape)
        ta = torch.from_numpy(a.astype(np.float32)).cuda()
        ta = ta.unsqueeze(0).repeat(8, 1, 1)

        a_idx = (a / a.max() * 480).astype("uint8")
        show_a = np.zeros((480, 480, 3), dtype="uint8")
        show_a[a_idx[:, 0], a_idx[:, 1], :] = np.array([255, 255, 255])

        b_ms = BatchMeanShiftTorch(bw, batch_size=1000)
        ctr, labels = b_ms.fit_batch(ta)
        for idx, lb in zip(a_idx, labels[0]):
            if lb.item():
                show_a[idx[0], idx[1], :] = np.array([255, 255, 0])
        ctr = (ctr[0].cpu().numpy() / a.max() * 480).astype("uint8")
        show_a = cv2.circle(show_a, (ctr[1], ctr[0]), 3, (255, 0, 0), -1)

        imshow('show_a', show_a)
        waitKey(0)


def main():
    # test()
    test_bs()


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
