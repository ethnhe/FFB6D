#!/usr/bin/env mdl
import os
import cv2
import math
import trimesh
import pyrender
import numpy as np
import pickle as pkl
from tqdm import tqdm
from scipy import stats
from glob import glob
import json
from utils import PoseUtils, MeshUtils, SysUtils, ImgPcldUtils
from fps.fps_utils import farthest_point_sampling


pose_utils = PoseUtils()
mesh_utils = MeshUtils()
sys_utils = SysUtils()
img_pcld_utils = ImgPcldUtils()


def rnder_one_scene(args, mesh_pth, obj_pose, camera_pose):
    try:
        fuze_trimesh = trimesh.load(mesh_pth)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    except Exception:
        print("Error loadding from {}".format(mesh_pth))
        return

    scene = pyrender.Scene(ambient_light=[0.9, 0.9, 0.9])
    nm = pyrender.Node(mesh=mesh, matrix=obj_pose)
    scene.add_node(nm)

    h, w = args.h, args.w
    if type(args.K) == list:
        K = np.array(args.K).reshape(3, 3)
    else:
        K = args.K
    camera = pyrender.IntrinsicsCamera(K[0][0], K[1][1], K[0][2], K[1][2])

    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    # light = pyrender.light.Light(color=[1.0, 1.0, 1.0], intensity=5)
    scene.add(light)
    r = pyrender.OffscreenRenderer(w, h)
    color, depth = r.render(scene)

    return color, depth


def extract_one_scene_textured_kp3ds(args, color, depth, o2c_pose, i_cam):
    if args.extractor == 'SIFT':
        extractor = cv2.xfeatures2d.SIFT_create()
    else:  # use orb
        extractor = cv2.ORB_create()

    if type(args.K) == list:
        K = np.array(args.K).reshape(3, 3)
    else:
        K = args.K

    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kps, des = extractor.detectAndCompute(gray, None)

    kp_xys = np.array([kp.pt for kp in kps]).astype(np.int32)
    if (kp_xys.shape[0] == 0):
        return None

    kp_idxs = (kp_xys[:, 1], kp_xys[:, 0])

    dpt_xyz = img_pcld_utils.dpt_2_cld(depth, 1.0, K)
    kp_x = dpt_xyz[:, :, 0][kp_idxs][..., None]
    kp_y = dpt_xyz[:, :, 1][kp_idxs][..., None]
    kp_z = dpt_xyz[:, :, 2][kp_idxs][..., None]
    kp_xyz = np.concatenate((kp_x, kp_y, kp_z), axis=1)

    # filter by dpt (pcld)
    kp_xyz, msk = img_pcld_utils.filter_pcld(kp_xyz)

    bgr_kp = bgr.copy()
    kps_2d = img_pcld_utils.project_p3ds(kp_xyz, cam_scale=1.0, K=K)
    bgr_kp = img_pcld_utils.draw_p2ds(bgr_kp, kps_2d, color=(0, 255, 0))

    # transform to object coordinate system
    kp_xyz = (kp_xyz - o2c_pose[:3, 3]).dot(o2c_pose[:3, :3])

    kps_3d = kp_xyz

    if args.vis:
        cv2.imshow("color_kp", bgr_kp)
        cv2.imshow("color", bgr)
        cv2.imshow("depth", depth)
        cmd = cv2.waitKey(0)
        if cmd == ord('q'):
            exit()

    return kps_3d


def extract_textured_kp3ds(args, mesh_pth, sv_kp=True):
    xyzs = mesh_utils.get_p3ds_from_mesh(mesh_pth, scale2m=args.scale2m)
    mean = np.mean(xyzs, axis=0)
    obj_pose = np.eye(4)
    obj_pose[:3, 3] = -1.0 * mean
    bbox = mesh_utils.get_3D_bbox(xyzs)
    r = mesh_utils.get_r(bbox)

    sph_r = r / 0.035 * 0.18
    positions = pose_utils.CameraPositions(
        args.n_longitude, args.n_latitude, sph_r
    )
    cam_poses = [pose_utils.getCameraPose(pos) for pos in positions]
    kp3ds = []
    for i_cam, cam_pose in enumerate(cam_poses):
        # 6D pose of object in cv camer coordinate system
        o2c_pose = pose_utils.get_o2c_pose_cv(cam_pose, obj_pose)
        # transform to object coordinate system
        color, depth = rnder_one_scene(args, mesh_pth, obj_pose, cam_pose)
        frame_kp3ds = extract_one_scene_textured_kp3ds(args, color, depth, o2c_pose, i_cam)
        if kp3ds is not None:
            kp3ds += list(frame_kp3ds)
        # pclds += list(data['dpt_pcld'])

    if sv_kp:
        with open("%s_%s_textured_kp3ds.obj" % (args.obj_name, args.extractor), 'w') as of:
            for p3d in kp3ds:
                print('v ', p3d[0], p3d[1], p3d[2], file=of)
    return kp3ds


# Select keypoint with Farthest Point Sampling (FPS) algorithm
def get_farthest_3d(p3ds, num=8, init_center=False):
    fps = farthest_point_sampling(p3ds, num, init_center=init_center)
    return fps


def test():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--obj_name", type=str, default="cracker_box",
        help="Object name."
    )
    parser.add_argument(
        "--obj_pth", type=str, default="example_mesh/003_cracker_box/textured.obj",
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

    kp3ds = extract_textured_kp3ds(args, args.obj_pth)
    textured_fps = get_farthest_3d(np.array(kp3ds), num=8, init_center=False)
    with open("%s_%s_textured_fps.obj" % (args.obj_name, args.extractor), 'w') as of:
        for p3d in textured_fps:
            print('v ', p3d[0], p3d[1], p3d[2], file=of)


if __name__ == "__main__":
    test()

# vim: ts=4 sw=4 sts=4 expandtab
