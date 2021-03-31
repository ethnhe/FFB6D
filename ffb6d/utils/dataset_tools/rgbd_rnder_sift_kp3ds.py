#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import ctypes as ct
import pickle as pkl

import random
from random import randint
from random import shuffle
from tqdm import tqdm
from scipy import stats
from glob import glob
from cv2 import imshow, waitKey
from utils import ImgPcldUtils, MeshUtils, PoseUtils, SysUtils


SO_P = './raster_triangle/rastertriangle_so.so'
RENDERER = np.ctypeslib.load_library(SO_P, '.')

mesh_utils = MeshUtils()
img_pcld_utils = ImgPcldUtils()
pose_utils = PoseUtils()
sys_utils = SysUtils()


def load_mesh_c(mdl_p, scale2m):
    if 'ply' in mdl_p:
        meshc = mesh_utils.load_ply_model(mdl_p, scale2m=scale2m)
    meshc['face'] = np.require(meshc['face'], 'int32', 'C')
    meshc['r'] = np.require(np.array(meshc['r']), 'float32', 'C')
    meshc['g'] = np.require(np.array(meshc['g']), 'float32', 'C')
    meshc['b'] = np.require(np.array(meshc['b']), 'float32', 'C')
    return meshc


def gen_one_zbuf_render(args, meshc, RT):
    if args.extractor == 'SIFT':
        extractor = cv2.xfeatures2d.SIFT_create()
    else:  # use orb
        extractor = cv2.ORB_create()

    h, w = args.h, args.w
    if type(args.K) == list:
        K = np.array(args.K).reshape(3, 3)
    R, T = RT[:3, :3], RT[:3, 3]

    new_xyz = meshc['xyz'].copy()
    new_xyz = np.dot(new_xyz, R.T) + T
    p2ds = np.dot(new_xyz.copy(), K.T)
    p2ds = p2ds[:, :2] / p2ds[:, 2:]
    p2ds = np.require(p2ds.flatten(), 'float32', 'C')

    zs = np.require(new_xyz[:, 2].copy(), 'float32', 'C')
    zbuf = np.require(np.zeros(h*w), 'float32', 'C')
    rbuf = np.require(np.zeros(h*w), 'int32', 'C')
    gbuf = np.require(np.zeros(h*w), 'int32', 'C')
    bbuf = np.require(np.zeros(h*w), 'int32', 'C')

    RENDERER.rgbzbuffer(
        ct.c_int(h),
        ct.c_int(w),
        p2ds.ctypes.data_as(ct.c_void_p),
        new_xyz.ctypes.data_as(ct.c_void_p),
        zs.ctypes.data_as(ct.c_void_p),
        meshc['r'].ctypes.data_as(ct.c_void_p),
        meshc['g'].ctypes.data_as(ct.c_void_p),
        meshc['b'].ctypes.data_as(ct.c_void_p),
        ct.c_int(meshc['n_face']),
        meshc['face'].ctypes.data_as(ct.c_void_p),
        zbuf.ctypes.data_as(ct.c_void_p),
        rbuf.ctypes.data_as(ct.c_void_p),
        gbuf.ctypes.data_as(ct.c_void_p),
        bbuf.ctypes.data_as(ct.c_void_p),
    )

    zbuf.resize((h, w))
    msk = (zbuf > 1e-8).astype('uint8')
    if len(np.where(msk.flatten() > 0)[0]) < 500:
        return None
    zbuf *= msk.astype(zbuf.dtype)  # * 1000.0

    bbuf.resize((h, w)), rbuf.resize((h, w)), gbuf.resize((h, w))
    bgr = np.concatenate((bbuf[:, :, None], gbuf[:, :, None], rbuf[:, :, None]), axis=2)
    bgr = bgr.astype('uint8')

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if args.vis:
        imshow("bgr", bgr.astype("uint8"))
        show_zbuf = zbuf.copy()
        min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
        show_zbuf[show_zbuf > 0] = (show_zbuf[show_zbuf > 0] - min_d) / (max_d - min_d) * 255
        show_zbuf = show_zbuf.astype(np.uint8)
        imshow("dpt", show_zbuf)
        show_msk = (msk / msk.max() * 255).astype("uint8")
        imshow("msk", show_msk)
        waitKey(0)

    data = {}
    data['depth'] = zbuf
    data['rgb'] = rgb
    data['mask'] = msk
    data['K'] = K
    data['RT'] = RT
    data['cls_typ'] = args.obj_name
    data['rnd_typ'] = 'render'

    kps, des = extractor.detectAndCompute(bgr, None)

    kp_xys = np.array([kp.pt for kp in kps]).astype(np.int32)
    kp_idxs = (kp_xys[:, 1], kp_xys[:, 0])

    dpt_xyz = img_pcld_utils.dpt_2_cld(zbuf, 1.0, K)
    kp_x = dpt_xyz[:, :, 0][kp_idxs][..., None]
    kp_y = dpt_xyz[:, :, 1][kp_idxs][..., None]
    kp_z = dpt_xyz[:, :, 2][kp_idxs][..., None]
    kp_xyz = np.concatenate((kp_x, kp_y, kp_z), axis=1)

    # filter by dpt (pcld)
    kp_xyz, msk = img_pcld_utils.filter_pcld(kp_xyz)
    kps = [kp for kp, valid in zip(kps, msk) if valid]  # kps[msk]
    des = des[msk, :]

    # 6D pose of object in cv camer coordinate system
    # transform to object coordinate system
    kp_xyz = (kp_xyz - RT[:3, 3]).dot(RT[:3, :3])
    dpt_xyz = dpt_xyz[dpt_xyz[:, :, 2] > 0, :]
    dpt_pcld = (dpt_xyz.reshape(-1, 3) - RT[:3, 3]).dot(RT[:3, :3])

    data['kp_xyz'] = kp_xyz
    data['dpt_pcld'] = dpt_pcld

    return data


def extract_textured_kp3ds(args, mesh_pth, sv_kp=False):
    meshc = load_mesh_c(mesh_pth, args.scale2m)
    xyzs = meshc['xyz']
    mean = np.mean(xyzs, axis=0)
    obj_pose = np.eye(4)
    # obj_pose[:3, 3] = -1.0 * mean
    bbox = mesh_utils.get_3D_bbox(xyzs)
    r = mesh_utils.get_r(bbox)
    print("r:", r)

    sph_r = r / 0.035 * 0.18
    positions = pose_utils.CameraPositions(
        args.n_longitude, args.n_latitude, sph_r
    )
    cam_poses = [pose_utils.getCameraPose(pos) for pos in positions]
    kp3ds = []
    # pclds = []
    for cam_pose in cam_poses:
        o2c_pose = pose_utils.get_o2c_pose_cv(cam_pose, obj_pose)
        # transform to object coordinate system
        data = gen_one_zbuf_render(args, meshc, o2c_pose)
        kp3ds += list(data['kp_xyz'])
        # pclds += list(data['dpt_pcld'])

    if sv_kp:
        with open("%s_%s_textured_kp3ds.obj" % (args.obj_name, args.extractor), 'w') as of:
            for p3d in kp3ds:
                print('v ', p3d[0], p3d[1], p3d[2], file=of)
    return kp3ds


def test():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--obj_name", type=str, default="ape",
        help="Object name."
    )
    parser.add_argument(
        "--ply_pth", type=str, default="example_mesh/ape.ply",
        help="path to object ply."
    )
    parser.add_argument(
        '--debug', action="store_true",
        help="To show the generated images or not."
    )
    parser.add_argument(
        '--vis', action="store_true",
        help="visulaize generated images."
    )
    parser.add_argument(
        '--h', type=int, default=480,
        help="height of rendered RGBD images."
    )
    parser.add_argument(
        '--w', type=int, default=640,
        help="width of rendered RGBD images."
    )
    parser.add_argument(
        '--K', type=int, default=[700, 0, 320, 0, 700, 240, 0, 0, 1],
        help="camera intrinsix."
    )
    parser.add_argument(
        '--scale2m', type=float, default=1.0,
        help="scale to transform unit of object to be in meter."
    )
    parser.add_argument(
        '--n_longitude', type=int, default=3,
        help="number of longitude on sphere to sample."
    )
    parser.add_argument(
        '--n_latitude', type=int, default=3,
        help="number of latitude on sphere to sample."
    )
    parser.add_argument(
        '--extractor', type=str, default="ORB",
        help="2D keypoint extractor, SIFTO or ORB"
    )
    parser.add_argument(
        '--textured_3dkps_fd', type=str, default="textured_3D_keypoints",
        help="folder to store textured 3D keypoints."
    )
    args = parser.parse_args()
    args.K = np.array(args.K).reshape(3, 3)

    kp3ds = extract_textured_kp3ds(args, args.ply_pth)


if __name__ == "__main__":
    test()
# vim: ts=4 sw=4 sts=4 expandtab
