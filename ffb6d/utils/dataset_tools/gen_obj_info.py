#!/usr/bin/env python3
import os
import numpy as np
import glob
from plyfile import PlyData
from tqdm import tqdm
import pickle as pkl
from fps.fps_utils import farthest_point_sampling
from argparse import ArgumentParser
from utils import MeshUtils, ImgPcldUtils, SysUtils


parser = ArgumentParser()
parser.add_argument(
    "--obj_name", type=str, default="ape", help="Object name."
)
parser.add_argument(
    "--obj_pth", type=str, default="example_mesh/ape.ply",
    help="path to object ply."
)
parser.add_argument(
    "--sv_fd", type=str, help="path to save the generated mesh info."
)
parser.add_argument(
    '--scale2m', type=float, default=1.0,
    help="scale to transform unit of object to be in meter."
)
parser.add_argument(
    '--vis', action="store_true", help="visulaize rendered images."
)
parser.add_argument(
    '--h', type=int, default=480, help="height of rendered RGBD images."
)
parser.add_argument(
    '--w', type=int, default=640, help="width of rendered RGBD images."
)
parser.add_argument(
    '--K', type=int, default=[700, 0, 320, 0, 700, 240, 0, 0, 1],
    help="camera intrinsix."
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
    '--n_keypoint', type=int, default=8,
    help="number of keypoints to extract."
)
parser.add_argument(
    '--textured_3dkps_fd', type=str, default="textured_3D_keypoints",
    help="folder to store textured 3D keypoints."
)
parser.add_argument(
    '--use_pyrender', action='store_true',
    help="use pyrender or raster_triangle"
)
parser.print_help()
args = parser.parse_args()

mesh_utils = MeshUtils()
sys_utils = SysUtils()

print(args)

if args.use_pyrender:
    from pyrender_sift_kp3ds import extract_textured_kp3ds
else:
    from rgbd_rnder_sift_kp3ds import extract_textured_kp3ds


# Read object vertexes from text file
def get_p3ds_from_txt(pxyz_pth):
    pointxyz = np.loadtxt(pxyz_pth, dtype=np.float32)
    return pointxyz


# Select keypoint with Farthest Point Sampling (FPS) algorithm
def get_farthest_3d(p3ds, num=8, init_center=False):
    fps = farthest_point_sampling(p3ds, num, init_center=init_center)
    return fps


# Compute and save all mesh info
def gen_one_mesh_info(args, obj_pth, sv_fd):
    sys_utils.ensure_dir(sv_fd)

    p3ds = mesh_utils.get_p3ds_from_mesh(obj_pth, scale2m=args.scale2m)

    c3ds = mesh_utils.get_3D_bbox(p3ds)
    c3ds_pth = os.path.join(sv_fd, "%s_corners.txt" % args.obj_name)
    with open(c3ds_pth, 'w') as of:
        for p3d in c3ds:
            print(p3d[0], p3d[1], p3d[2], file=of)

    radius = mesh_utils.get_r(c3ds)
    r_pth = os.path.join(sv_fd, "%s_radius.txt" % args.obj_name)
    with open(r_pth, 'w') as of:
        print(radius, file=of)

    ctr = mesh_utils.get_centers_3d(c3ds)
    ctr_pth = os.path.join(sv_fd, "%s_center.txt" % args.obj_name)
    with open(ctr_pth, 'w') as of:
        print(ctr[0], ctr[1], ctr[2], file=of)

    fps = get_farthest_3d(p3ds, num=args.n_keypoint)
    fps_pth = os.path.join(sv_fd, "%s_fps.txt" % args.obj_name)
    with open(fps_pth, 'w') as of:
        for p3d in fps:
            print(p3d[0], p3d[1], p3d[2], file=of)

    textured_kp3ds = np.array(extract_textured_kp3ds(args, args.obj_pth))
    print(p3ds.shape, textured_kp3ds.shape)
    textured_fps = get_farthest_3d(textured_kp3ds, num=args.n_keypoint)
    textured_fps_pth = os.path.join(sv_fd, "%s_%s_fps.txt" % (args.obj_name, args.extractor))
    with open(textured_fps_pth, 'w') as of:
        for p3d in textured_fps:
            print(p3d[0], p3d[1], p3d[2], file=of)


def main():
    gen_one_mesh_info(args, args.obj_pth, args.sv_fd)


if __name__ == "__main__":
    main()


# vim: ts=4 sw=4 sts=4 expandtab
