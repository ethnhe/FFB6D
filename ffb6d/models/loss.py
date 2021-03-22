import time
import numpy as np
import torch.nn as nn
import random
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable


class FocalLoss(_Loss):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            # print("fcls input.size", input.size(), target.size())
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        # print("fcls reshape input.size", input.size(), target.size())

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def of_l1_loss(
        pred_ofsts, kp_targ_ofst, labels,
        sigma=1.0, normalize=True, reduce=False
):
    '''
    :param pred_ofsts:      [bs, n_kpts, n_pts, c]
    :param kp_targ_ofst:    [bs, n_pts, n_kpts, c]
    :param labels:          [bs, n_pts, 1]
    '''
    w = (labels > 1e-8).float()
    bs, n_kpts, n_pts, c = pred_ofsts.size()
    sigma_2 = sigma ** 3
    w = w.view(bs, 1, n_pts, 1).repeat(1, n_kpts, 1, 1).contiguous()
    kp_targ_ofst = kp_targ_ofst.view(bs, n_pts, n_kpts, c)
    kp_targ_ofst = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()
    diff = pred_ofsts - kp_targ_ofst
    abs_diff = torch.abs(diff)
    abs_diff = w * abs_diff
    in_loss = abs_diff

    if normalize:
        in_loss = torch.sum(
            in_loss.view(bs, n_kpts, -1), 2
        ) / (torch.sum(w.view(bs, n_kpts, -1), 2) + 1e-3)

    if reduce:
        in_loss = torch.mean(in_loss)

    return in_loss


class OFLoss(_Loss):
    def __init__(self):
        super(OFLoss, self).__init__(True)

    def forward(
        self, pred_ofsts, kp_targ_ofst, labels,
        normalize=True, reduce=False
    ):
        l1_loss = of_l1_loss(
            pred_ofsts, kp_targ_ofst, labels,
            sigma=1.0, normalize=True, reduce=False
        )

        return l1_loss


class CosLoss(_Loss):
    def __init__(self, eps=1e-5):
        super(CosLoss, self).__init__(True)
        self.eps = eps

    def forward(
        self, pred_ofsts, kp_targ_ofst, labels, normalize=True
    ):
        '''
        :param pred_ofsts:      [bs, n_kpts, n_pts, c]
        :param kp_targ_ofst:    [bs, n_pts, n_kpts, c]
        :param labels:          [bs, n_pts, 1]
        '''
        print("pred size", pred_ofsts.size(), kp_targ_ofst.size())
        w = (labels > 1e-8).float()
        bs, n_kpts, n_pts, c = pred_ofsts.size()
        pred_vec = pred_ofsts / (torch.norm(pred_ofsts, dim=3, keepdim=True) + self.eps)
        print("pred_ofsts: ", pred_ofsts, "pred_vec:", pred_vec)
        w = w.view(bs, 1, n_pts, 1).repeat(1, n_kpts, 1, 1).contiguous()
        kp_targ_ofst = kp_targ_ofst.view(bs, n_pts, n_kpts, 3)
        kp_targ_ofst = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()
        targ_vec = kp_targ_ofst / (torch.norm(kp_targ_ofst, dim=3, keepdim=True) + self.eps)

        cos_sim = pred_vec * targ_vec
        in_loss = -1.0 * w * cos_sim

        if normalize:
            in_loss = torch.sum(
                in_loss.view(bs, n_kpts, -1), 2
            ) / (torch.sum(w.view(bs, n_kpts, -1), 2) + 1e-3)

        return in_loss


###### LOSSES #######

class BerHuLoss(nn.Module):
    def __init__(self, scale=0.5, eps=1e-5):
        super(BerHuLoss, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        img1 = img1[img2 > self.eps]
        img2 = img2[img2 > self.eps]

        diff = torch.abs(img1 - img2)
        threshold = self.scale * torch.max(diff).detach()
        mask = diff > threshold
        diff[mask] = ((img1[mask]-img2[mask])**2 + threshold**2) / (2*threshold + self.eps)
        return diff.sum() / diff.numel()


class LogDepthL1Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(LogDepthL1Loss, self).__init__()
        self.eps = eps
    def forward(self, pred, gt):
        pred = pred.view(-1)
        gt = gt.view(-1)
        mask = gt > self.eps
        diff = torch.abs(torch.log(gt[mask]) - pred[mask])
        return diff.mean()


class PcldSmoothL1Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        xmap = np.array([[j for i in range(640)] for j in range(480)])

    def dpt_2_pcld(self, dpt, K, xmap, ymap, scale2m=1.0):
        bs, _, h, w = dpt.size()
        dpt = dpt.view(bs, h, w).contiguous()

        dpt = dpt * scale2m
        cx = K[:, 0, 2].view(bs, 1, 1).repeat(1, h, w)
        cy = K[:, 1, 2].view(bs, 1, 1).repeat(1, h, w)
        fx = K[:, 0, 0].view(bs, 1, 1).repeat(1, h, w)
        fy = K[:, 1, 1].view(bs, 1, 1).repeat(1, h, w)
        row = (ymap - cx) * dpt / fx
        col = (xmap - cy) * dpt / fy
        # row = (ymap - K[:, 0, 2]) * dpt / K[0][0]
        # col = (xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = torch.cat([
            row.unsqueeze(-1), col.unsqueeze(-1), dpt.unsqueeze(-1)
        ], dim=3)
        return dpt_3d

    def forward(self, pred, gt, K, xmap, ymap):
        bs = pred.size()[0]
        pred_pcld = self.dpt_2_pcld(pred, K, xmap, ymap)

        img1 = torch.zeros_like(pred_pcld)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred_pcld)
        img2 = img2.copy_(gt)

        msk = img2[:, :, :, 2] > self.eps
        # img1[~msk, :] = 0.
        # img2[~msk, :] = 0.
        loss = nn.SmoothL1Loss(reduction="sum")(img1[msk, :], img2[msk, :])
        loss = loss / msk.float().sum() * bs
        return loss


###### METRICS #######

class DepthL1Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        bs = pred.size()[0]
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        mask = gt > self.eps
        img1[~mask] = 0.
        img2[~mask] = 0.
        # return nn.L1Loss(reduction="sum")(img1, img2), pred.numel()
        loss = nn.L1Loss(reduction="sum")(img1, img2)
        loss = loss / mask.float().sum() * bs
        return loss


class DepthL2Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DepthL2Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt):
        img1 = torch.zeros_like(pred)
        img2 = torch.zeros_like(gt)

        img1 = img1.copy_(pred)
        img2 = img2.copy_(gt)

        mask = gt > self.eps
        img1[~mask] = 0.
        img2[~mask] = 0.
        return nn.MSELoss(reduction="sum")(img1, img2), pred.numel()


class OfstMapL1Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, rgb_labels, pred, gt, normalize=True, reduce=True):
        wgt = (rgb_labels > 1e-8).float()
        bs, n_kpts, c, h, w = pred.size()
        wgt = wgt.view(bs, 1, 1, h, w).repeat(1, n_kpts, c, 1, 1).contiguous()

        diff = pred - gt
        abs_diff = torch.abs(diff)
        abs_diff = wgt * abs_diff
        in_loss = abs_diff

        if normalize:
            in_loss = torch.sum(
                in_loss.view(bs, n_kpts, -1), 2
            ) / (torch.sum(wgt.view(bs, n_kpts, -1), 2) + 1e-3)

        if reduce:
            in_loss = torch.mean(in_loss)

        return in_loss


class OfstMapKp3dL1Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def to_p3ds(self, xs, ys, zs, Ki):
        bs, n_kpts, h, w = xs.size()
        p3ds = torch.zeros_like(zs)
        p3ds = p3ds.view(bs, n_kpts, 1, h, w).repeat(1, 1, 3, 1, 1)
        for i in range(n_kpts):
            x, y, z = xs[:, i, :, :], ys[:, i, :, :], zs[:, i, :, :]
            x = x.reshape(bs, 1, -1)
            y = y.reshape(bs, 1, -1)
            z = z.reshape(bs, 1, -1)
            x = x * z
            y = y * z
            xyz = torch.cat((x, y, z), dim=1)
            xyz = torch.bmm(Ki, xyz)
            xyz = xyz.reshape(bs, 3, h, w)
            p3ds[:, i, :, :, :] = xyz
        return p3ds

    def forward(
        self, rgb_labels, pred, xmap, ymap, kp3d_map, Ki, normalize=True, reduce=False,
        scale=1
    ):
        """
            rgb_labels: [bs, h, w]
            pred:       [bs, n_kpts, c, h, w]
            xmap, ymap: [bs, h, w]
            kp3d_map:   [bs, n_kpts, c, h, w]
            Ki:          [bs, 3, 3]
        """
        wgt = (rgb_labels > 1e-8).half()
        if kp3d_map.dtype != torch.half:
            wgt = wgt.float()
        bs, n_kpts, c, h, w = pred.size()
        xmap = xmap.view(bs, 1, h, w).repeat(1, n_kpts, 1, 1)
        ymap = ymap.view(bs, 1, h, w).repeat(1, n_kpts, 1, 1)
        ys = (xmap - pred[:, :, 0, :, :]) * scale
        xs = (ymap - pred[:, :, 1, :, :]) * scale
        zs = pred[:, :, 2, :, :]

        pred_kp3d_map = self.to_p3ds(xs, ys, zs, Ki)

        wgt = wgt.view(bs, 1, 1, h, w).repeat(1, n_kpts, c, 1, 1).contiguous()

        diff = pred_kp3d_map - kp3d_map
        abs_diff = torch.abs(diff)
        abs_diff = wgt * abs_diff
        in_loss = abs_diff

        if normalize:
            in_loss = torch.sum(
                in_loss.view(bs, n_kpts, -1), 2
            ) / (torch.sum(wgt.view(bs, n_kpts, -1), 2) + 1e-3)

        if reduce:
            in_loss = torch.mean(in_loss)

        return in_loss
