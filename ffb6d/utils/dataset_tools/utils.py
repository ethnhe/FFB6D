#!/usr/bin/env mdl
import os
import cv2
import math
import numpy as np
from scipy import stats
from plyfile import PlyData


class SysUtils():
    def __init__(self):
        pass

    def ensure_dir(self, pth):
        if not os.path.exists(pth):
            os.system("mkdir -p {}".format(pth))


class MeshUtils():
    def __init__(self):
        pass

    def get_p3ds_from_obj(self, pth, scale2m=1.):
        xyz_lst = []
        with open(pth, 'r') as f:
            for line in f.readlines():
                if 'v ' not in line or line[0] != 'v':
                    continue
                xyz_str = [
                    item.strip() for item in line.split(' ')
                    if len(item.strip()) > 0 and 'v' not in item
                ]
                xyz = np.array(xyz_str[0:3]).astype(np.float)
                xyz_lst.append(xyz)
        return np.array(xyz_lst) / scale2m

    def load_ply_model(self, model_path, scale2m=1., ret_dict=True):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        r = data['red']
        g = data['green']
        b = data['blue']
        face_raw = ply.elements[1].data
        face = []
        for item in face_raw:
            face.append(item[0])

        n_face = len(face)
        face = np.array(face).flatten()
        n_pts = len(x)

        xyz = np.stack([x, y, z], axis=-1) / scale2m
        if not ret_dict:
            return n_pts, xyz, r, g, b, n_face, face
        else:
            ret_dict = dict(
                n_pts=n_pts, xyz=xyz, r=r, g=g, b=b, n_face=n_face, face=face
            )
            return ret_dict

    # Read object vertexes from ply file
    def get_p3ds_from_ply(self, ply_pth, scale2m=1.):
        print("loading p3ds from ply:", ply_pth)
        ply = PlyData.read(ply_pth)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        p3ds = np.stack([x, y, z], axis=-1)
        p3ds = p3ds / float(scale2m)
        print("finish loading ply.")
        return p3ds

    def get_p3ds_from_mesh(self, mesh_pth, scale2m=1.0):
        if '.ply' in mesh_pth:
            return self.get_p3ds_from_ply(mesh_pth, scale2m=scale2m)
        else:
            return self.get_p3ds_from_obj(mesh_pth, scale2m=scale2m)

    # Read object vertexes from text file
    def get_p3ds_from_txt(self, pxyz_pth):
        pointxyz = np.loadtxt(pxyz_pth, dtype=np.float32)
        return pointxyz

    # Compute the 3D bounding box from object vertexes
    def get_3D_bbox(self, pcld, small=False):
        min_x, max_x = pcld[:, 0].min(), pcld[:, 0].max()
        min_y, max_y = pcld[:, 1].min(), pcld[:, 1].max()
        min_z, max_z = pcld[:, 2].min(), pcld[:, 2].max()
        bbox = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        if small:
            center = np.mean(bbox, 0)
            bbox = (bbox - center[None, :]) * 2.0 / 3.0 + center[None, :]
        return bbox

    # Compute the radius of object
    def get_r(self, bbox):
        return np.linalg.norm(bbox[7,:] - bbox[0,:]) / 2.0

    # Compute the center of object
    def get_centers_3d(self, corners_3d):
        centers_3d = (np.max(corners_3d, 0) + np.min(corners_3d, 0)) / 2
        return centers_3d



class ImgPcldUtils():
    def __init__(self):
        pass

    def draw_p2ds(self, img, p2ds, r=1, color=(255, 0, 0)):
        h, w = img.shape[0], img.shape[1]
        p2ds = p2ds.astype(np.int32)
        for p2d in p2ds:
            p2d[0] = np.clip(p2d[0], 0, w)
            p2d[1] = np.clip(p2d[1], 0, h)
            img = cv2.circle(
                img, (p2d[0], p2d[1]), r, color, -1
            )
        return img

    def project_p3ds(self, p3d, cam_scale=1000.0, K=None):
        p3d = p3d * cam_scale
        p2d = np.dot(p3d, K.T)
        p2d_3 = p2d[:, 2]
        p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
        p2d[:, 2] = p2d_3
        p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
        return p2d

    def dpt_2_cld(self, dpt, cam_scale, K):
        h, w = dpt.shape[0], dpt.shape[1]
        xmap = np.array([[j for i in range(w)] for j in range(h)])
        ymap = np.array([[i for i in range(w)] for j in range(h)])

        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        msk_dp = dpt > 1e-6
        choose = msk_dp.flatten()
        choose[:] = True
        if len(choose) < 1:
            return None, None

        dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        pt2 = dpt_mskd / cam_scale
        cam_cx, cam_cy = K[0][2], K[1][2]
        cam_fx, cam_fy = K[0][0], K[1][1]
        pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
        cld = np.concatenate((pt0, pt1, pt2), axis=1)
        cld = cld.reshape(h, w, 3)
        return cld

    def K_dpt_2_cld(self, dpt, cam_scale, K):
        dpt = dpt.astype(np.float32)
        dpt /= cam_scale

        Kinv = np.linalg.inv(K)

        h, w = dpt.shape[0], dpt.shape[1]

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h, w), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(w*h, 3)

        # backproj
        R = np.dot(Kinv, x2d.transpose())

        # compute 3D points
        X = R * np.tile(dpt.reshape(1, w*h), (3, 1))
        X = np.array(X).transpose()

        X = X.reshape(h, w, 3)
        # good_msk = np.isfinite(X[:, :, 0]) and X[:, :, 2] > 0
        # print(good_msk.shape, X.shape)
        # dpt[~good_msk, :] = 0.0
        return X

    def filter_pcld(self, pcld):
        """
            pcld: [N, c] point cloud.
        """
        if len(pcld.shape) > 2:
            pcld = pcld.reshape(-1, 3)
        msk1 = np.isfinite(pcld[:, 0])
        msk2 = pcld[:, 2] > 1e-8
        msk = msk1 & msk2
        pcld = pcld[msk, :]
        return pcld, msk


class PoseUtils():
    def __init__(self):
        pass

    # Returns camera rotation and translation matrices from OpenGL.
    #
    # There are 3 coordinate systems involved:
    #    1. The World coordinates: "world"
    #       - right-handed
    #    2. The Blender camera coordinates: "bcam"
    #       - x is horizontal
    #       - y is up
    #       - right-handed: negative z look-at direction
    #    3. The desired computer vision camera coordinates: "cv"
    #       - x is horizontal
    #       - y is down (to align to the actual pixel coordinates
    #         used in digital images)
    #       - right-handed: positive z look-at direction
    def get_3x4_RT_matrix_from_blender(self, camera_pose):
        # bcam stands for blender camera
        R_bcam2cv = np.array([(1, 0, 0), (0, -1, 0), (0, 0, -1)])

        # Use matrix_world instead to account for all constraints
        location, rotation = camera_pose[:3, 3], camera_pose[:3,:3]
        R_world2bcam = rotation.T

        # Convert camera location to translation vector used in coordinate changes
        # Use location from matrix_world to account for constraints:
        T_world2bcam = -1 * R_world2bcam @ location

        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = R_bcam2cv @ R_world2bcam
        T_world2cv = R_bcam2cv @ T_world2bcam

        # put into 4x4 matrix
        RT = np.eye(4)
        RT[:3,:3] = R_world2cv
        RT[:3, 3] = T_world2cv
        return RT

    def get_o2c_pose_cv(self, cam_pose, obj_pose):
        """
        Get object 6D pose in cv camera coordinate system
        cam_pose: camera rotation and translation matrices from get_3x4_RT_matrix_from_blender().
        obj_pose: obj_pose in world coordinate system
        """
        w2c = self.get_3x4_RT_matrix_from_blender(cam_pose)
        o2c = np.matmul(w2c, obj_pose)
        return o2c

    def isRotationMatrix(self, R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R) :

        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    # Calculates Rotation Matrix given euler angles.
    def eulerAnglesToRotationMatrix(self, theta) :
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R
    # vim: ts=4 sw=4 sts=4 expandtab

    def sample_sphere(self, num_samples, cls='ape'):
        """ sample angles from the sphere
        reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        """
        flat_objects = ['037_scissors', '051_large_clamp', '052_extra_large_clamp']
        if cls in flat_objects:
            begin_elevation = 30
        else:
            begin_elevation = 0
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        azimuths = []
        elevations = []
        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            azimuths.append(np.rad2deg(2 * np.pi * n * phi % (2 * np.pi)))
            elevations.append(np.rad2deg(np.arcsin(z)))
        return np.array(azimuths), np.array(elevations)

    def sample_poses(self, num_samples):
        s = np.sqrt(2) / 2
        cam_pose = np.array([
            [0.0, -s, s, 0.50],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, s, s, 0.60],
            [0.0, 0.0, 0.0, 1.0],
        ])
        eulers = self.rotationMatrixToEulerAngles(cam_pose[:3, :3]).reshape(1, -1)
        eulers = np.repeat(eulers, num_samples, axis=0)
        translations = cam_pose[:3, 3].reshape(1, 3)
        translations = np.repeat(translations, num_samples, axis=0)
        # print(eulers.shape, translations.shape)

        azimuths, elevations = self.sample_sphere(num_samples)
        # euler_sampler = stats.gaussian_kde(eulers.T)
        # eulers = euler_sampler.resample(num_samples).T
        eulers[:, 0] = azimuths
        eulers[:, 1] = elevations
        # translation_sampler = stats.gaussian_kde(translations.T)
        # translations = translation_sampler.resample(num_samples).T
        RTs = []
        for euler in eulers:
            RTs.append(self.eulerAnglesToRotationMatrix(euler))
        RTs = np.array(RTs)
        # print(eulers.shape, translations.shape, RTs.shape)
        return RTs, translations
        # np.save(self.blender_poses_path, np.concatenate([eulers, translations], axis=-1))

    def CameraPositions(self, n1, n2, r):
        'sample on a ball'
        theta_list = np.linspace(0.1, np.pi, n1+1)
        theta_list = theta_list[:-1]
        phi_list = np.linspace(0, 2*np.pi, n2+1)
        phi_list = phi_list[:-1]

        def product(a_lst, b_lst):
            res_lst = []
            for a in a_lst:
                for b in b_lst:
                    res_lst.append((a, b))
            return res_lst

        cpList = product(theta_list, phi_list)
        PositionList = []
        for theta, phi in cpList:
            x = r*math.sin(theta)*math.cos(phi)
            y = r*math.sin(theta)*math.sin(phi)
            z = r*math.cos(theta)
            PositionList.append((x, y, z))
        return PositionList

    def getCameraPose(self, T):
        '''
        OpenGL camera coordinates, the camera z-axis points away from the scene, the x-axis points right in image space, and the y-axis points up in image space.
        see https://pyrender.readthedocs.io/en/latest/examples/cameras.html
        '''
        z_direct = np.array(T)
        z_direct = z_direct/np.linalg.norm(z_direct)
        g_direct = np.array([0, 0, 1])
        x_direct = -np.cross(z_direct, g_direct)
        x_direct = x_direct/np.linalg.norm(x_direct)
        y_direct = np.cross(z_direct, x_direct)
        y_direct = y_direct/np.linalg.norm(y_direct)

        pose = np.array([x_direct, y_direct, z_direct])
        pose = np.transpose(pose)

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = pose
        camera_pose[:3, 3] = T
        return camera_pose

