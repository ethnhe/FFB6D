#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from datasets.ycb.ycb_dataset import Dataset as YCB_Dataset
from datasets.linemod.linemod_dataset import Dataset as LM_Dataset
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-dataset", type=str, default="linemod",
    help="Target dataset, ycb or linemod. (linemod as default)."
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
parser.add_argument(
    "-show", action='store_true', help="View from imshow or not."
)
args = parser.parse_args()

if args.dataset == "ycb":
    config = Config(ds_name=args.dataset)
else:
    config = Config(ds_name=args.dataset, cls_type=args.cls)
bs_utils = Basic_Utils(config)


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint['model_state']
        if 'module' in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec


def cal_view_pred_pose(model, data, epoch=0, obj_id=-1):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)
        _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        if args.dataset == "ycb":
            pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, True,
                None, None
            )
        else:
            pred_pose_lst = cal_frame_poses_lm(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, False, obj_id
            )
            pred_cls_ids = np.array([[1]])

        np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        if args.dataset == "ycb":
            np_rgb = np_rgb[:, :, ::-1].copy()
        ori_rgb = np_rgb.copy()
        for cls_id in cu_dt['cls_ids'][0].cpu().numpy():
            idx = np.where(pred_cls_ids == cls_id)[0]
            if len(idx) == 0:
                continue
            pose = pred_pose_lst[idx[0]]
            if args.dataset == "ycb":
                obj_id = int(cls_id[0])
            mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type=args.dataset).copy()
            mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
            if args.dataset == "ycb":
                K = config.intrinsic_matrix["ycb_K1"]
            else:
                K = config.intrinsic_matrix["linemod"]
            mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
            color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
            np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
        vis_dir = os.path.join(config.log_eval_dir, "pose_vis")
        ensure_fd(vis_dir)
        f_pth = os.path.join(vis_dir, "{}.jpg".format(epoch))
        if args.dataset == 'ycb':
            bgr = np_rgb
            ori_bgr = ori_rgb
        else:
            bgr = np_rgb[:, :, ::-1]
            ori_bgr = ori_rgb[:, :, ::-1]
        cv2.imwrite(f_pth, bgr)
        if args.show:
            imshow("projected_pose_rgb", bgr)
            imshow("original_rgb", ori_bgr)
            waitKey()
    if epoch == 0:
        print("\n\nResults saved in {}".format(vis_dir))


def main():
    if args.dataset == "ycb":
        test_ds = YCB_Dataset('test')
        obj_id = -1
    else:
        test_ds = LM_Dataset('test', cls_type=args.cls)
        obj_id = config.lm_obj_dict[args.cls]
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=20
    )

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints
    )
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )

    for i, data in tqdm.tqdm(
        enumerate(test_loader), leave=False, desc="val"
    ):
        cal_view_pred_pose(model, data, epoch=i, obj_id=obj_id)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
