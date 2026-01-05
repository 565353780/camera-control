import os
import torch
import numpy as np
import open3d as o3d
from copy import deepcopy
from typing import Union, Optional

from camera_control.Method.data import toNumpy, toTensor
from camera_control.Method.path import removeFile, createFileFolder


class CameraData(object):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        pos: Union[torch.Tensor, np.ndarray, list] = [0, 0, 0],
        look_at: Union[torch.Tensor, np.ndarray, list] = [1, 0, 0],
        up: Union[torch.Tensor, np.ndarray, list] = [0, 0, 1],
        world2camera: Union[torch.Tensor, np.ndarray, list, None] = None,
        dtype=torch.float32,
        device: str = 'cpu',
    ) -> None:
        """
        相机坐标系定义：
        - X轴：向右
        - Y轴：向上
        - Z轴：向后（相机看向 -Z 方向）
        - 图像在 XOY 平面上

        UV坐标系定义：
        - 原点在 (0, 0)，即图像左下角
        - u 方向：沿 X 轴（向右）
        - v 方向：沿 Y 轴（向上）
        """
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        if cx is None:
            self.cx = 0.5 * self.width
        else:
            self.cx = float(cx)
        if cy is None:
            self.cy = 0.5 * self.height
        else:
            self.cy = float(cy)
        self.dtype = dtype
        self.device = device

        if world2camera is not None:
            self.world2camera = toTensor(world2camera, self.dtype, self.device)
        else:
            pos = toTensor(pos)
            self.setWorld2Camera(look_at, up, pos)
        return

    @classmethod
    def fromDict(
        cls,
        data_dict: dict,
        dtype=torch.float32,
        device: str='cpu',
    ) -> "CameraData":
        camera = cls(
            width=data_dict['width'],
            height=data_dict['height'],
            fx=data_dict['fx'],
            fy=data_dict['fy'],
            cx=data_dict['cx'],
            cy=data_dict['cy'],
            world2camera=data_dict['world2camera'],
            dtype=dtype,
            device=device,
        )
        return camera

    @classmethod
    def fromDictFile(
        cls,
        npy_file_path: str,
        dtype=torch.float32,
        device: str='cpu',
    ) -> "CameraData":
        if not os.path.exists(npy_file_path):
            print('[ERROR][CameraData::fromDictFile]')
            print('\t npy file not exist!')
            print("\t npy_file_path:", npy_file_path)
            return cls()

        data_dict = np.load(npy_file_path, allow_pickle=True)

        return cls.fromDict(data_dict, dtype, device)

    def clone(self):
        return deepcopy(self)

    @property
    def intrinsic(self) -> torch.Tensor:
        return torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=self.dtype, device=self.device)

    @property
    def R(self) -> torch.Tensor:
        """从 world2camera 矩阵中提取旋转矩阵（世界到相机）"""
        return self.world2camera[:3, :3]

    @property
    def t(self) -> torch.Tensor:
        """从 world2camera 矩阵中提取平移矩阵（世界到相机）"""
        return self.world2camera[:3, 3]

    @property
    def pos(self) -> torch.Tensor:
        """从 world2camera 矩阵中提取相机位置（世界坐标系中）"""
        return -self.R.T @ self.t

    @property
    def world2cameraCV(self) -> torch.Tensor:
        C = torch.diag(torch.tensor([1, -1, -1, 1], dtype=self.dtype, device=self.device))
        return C @ self.world2camera @ C

    @property
    def camera2world(self) -> torch.Tensor:
        """
        计算相机坐标系到世界坐标系的变换矩阵

        正确的计算方式：
        world2camera = [R | t]    其中 R 是旋转矩阵，t 是平移向量
                       [0 | 1]

        camera2world = [R^T | -R^T @ t]
                       [0   | 1       ]
        """
        R_T = self.R.T

        camera2world = torch.eye(4, dtype=self.dtype, device=self.device)
        camera2world[:3, :3] = R_T  # R的转置
        camera2world[:3, 3] = -R_T @ self.t  # -R^T @ t

        return camera2world

    @property
    def camera2worldCV(self) -> torch.Tensor:
        C = torch.diag(torch.tensor([1, -1, -1, 1], dtype=self.dtype, device=self.device))
        return C @ self.camera2world @ C

    def setWorld2CameraByRt(
        self,
        R: Union[torch.Tensor, np.ndarray, list],
        t: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        R = toTensor(R, self.dtype, self.device).reshape(3, 3)
        t = toTensor(t, self.dtype, self.device).reshape(3)

        self.world2camera = torch.eye(4, dtype=self.dtype, device=self.device)
        self.world2camera[:3, :3] = R
        self.world2camera[:3, 3] = t
        return True

    def setWorld2Camera(
        self,
        look_at: Union[torch.Tensor, np.ndarray, list],
        up: Union[torch.Tensor, np.ndarray, list] = [0, 0, 1],
        pos: Union[torch.Tensor, np.ndarray, list, None] = None,
    ) -> bool:
        """
        设置世界坐标系到相机坐标系的变换矩阵

        相机坐标系定义：
        - X轴：向右
        - Y轴：向上
        - Z轴：向后（相机看向 -Z 方向）

        Args:
            look_at: 相机看向的世界坐标点
            up: 世界坐标系中的上方向向量
            pos: 相机在世界坐标系中的位置
        """
        look_at = toTensor(look_at, self.dtype, self.device)
        up = toTensor(up, self.dtype, self.device)

        if pos is None:
            pos = self.pos
        else:
            pos = toTensor(pos, self.dtype, self.device)

        look_at = look_at.flatten()
        up = up.flatten()
        pos = pos.flatten()

        # 计算相机的朝向（Z轴指向后方，所以相机看向的方向是 -Z）
        forward_world = look_at - pos
        forward_world = forward_world / (torch.linalg.norm(forward_world) + 1e-8)

        # 计算相机坐标系的 Z 轴（后方）在世界坐标系中的方向
        # Z轴 = -forward（因为相机看向 -Z）
        z_axis_world = -forward_world

        # 计算相机坐标系的 X 轴（右方向）
        # forward 叉乘 up 得到右方向
        up_normalized = up / (torch.linalg.norm(up) + 1e-8)
        x_axis_world = torch.linalg.cross(forward_world, up_normalized)
        x_axis_world = x_axis_world / (torch.linalg.norm(x_axis_world) + 1e-8)

        # 计算相机坐标系的 Y 轴（上方向）
        # z_axis 叉乘 x_axis 得到 y_axis
        # 或者 x_axis 叉乘 z_axis 得到 -y_axis
        # 使用右手系：x × y = z，所以 y = z × x
        y_axis_world = torch.linalg.cross(z_axis_world, x_axis_world)
        y_axis_world = y_axis_world / (torch.linalg.norm(y_axis_world) + 1e-8)

        # 构建旋转矩阵 R（世界到相机）
        # R 的每一行是相机坐标系的一个轴在世界坐标系中的表示
        R = torch.stack([x_axis_world, y_axis_world, z_axis_world], dim=0)

        # 构建平移向量 t
        t = -R @ pos

        self.setWorld2CameraByRt(R, t)
        return True

    def setWorld2CameraByCamera2WorldCV(
        self,
        camera2world_cv: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        """
        通过 OpenCV 坐标系下的 camera2world 矩阵设置 world2camera

        转换关系：
        camera2worldCV = C @ camera2world @ C
        其中 C = diag([1, -1, -1, 1])

        反推步骤：
        1. camera2world = C @ camera2worldCV @ C  (因为 C @ C = I)
        2. 从 camera2world 提取 R^T 和 -R^T @ t
        3. 计算 R = (R^T)^T 和 t = -R @ (camera2world[:3, 3])
        4. 构建 world2camera = [R | t; 0 | 1]

        Args:
            camera2world_cv: OpenCV 坐标系下的 camera2world 矩阵 (4x4)
        """
        camera2world_cv = toTensor(camera2world_cv, self.dtype, self.device).reshape(4, 4)

        # 定义坐标系转换矩阵
        C = torch.diag(torch.tensor([1, -1, -1, 1], dtype=self.dtype, device=self.device))

        # 从 OpenCV 坐标系转换到原始坐标系
        # camera2world = C @ camera2worldCV @ C
        camera2world = C @ camera2world_cv @ C

        # 从 camera2world 解析出 world2camera
        # camera2world = [R^T | -R^T @ t]
        #                [0   | 1       ]
        # 
        # world2camera = [R | t]
        #                [0 | 1]
        R_T = camera2world[:3, :3]  # 提取 R^T
        neg_RT_t = camera2world[:3, 3]  # 提取 -R^T @ t

        # 计算 R 和 t
        R = R_T.T  # R = (R^T)^T
        t = -R @ neg_RT_t  # t = -R @ (-R^T @ t) = R @ R^T @ t

        # 使用解析方法设置 world2camera
        self.setWorld2CameraByRt(R, t)

        return True

    def loadVGGTCameraFile(
        self,
        vggt_camera_txt_file_path: str,
    ) -> bool:
        if not os.path.exists(vggt_camera_txt_file_path):
            print('[ERROR][Camera::loadVGGTCameraFile]')
            print('\t vggt camera txt file not exist!')
            print('\t vggt_camera_txt_file_path:', vggt_camera_txt_file_path)
            return False

        camera2world_cv = np.eye(4)
        with open(vggt_camera_txt_file_path, 'r') as f:
            lines = f.readlines()

        for i in range(3):
            camera2world_cv[i] = [float(d) for d in lines[i].split()]

        self.setWorld2CameraByCamera2WorldCV(camera2world_cv)

        self.fx, _, _ = [float(d) for d in lines[3].split()]
        _, self.fy, _ = [float(d) for d in lines[4].split()]
        return True

    def focusOnPoints(
        self,
        points: Union[torch.Tensor, np.ndarray, list],
        view_ratio: float = 0.95,
    ) -> bool:
        points = toTensor(points, self.dtype, self.device)

        min_bound = torch.min(points, dim=0).values
        max_bound = torch.max(points, dim=0).values
        center = (min_bound + max_bound) / 2.0

        radius = torch.max(torch.linalg.norm(points - center, dim=1)).item()

        if self.width < self.height:
            f = self.fx
            target_px = self.width * view_ratio
        else:
            f = self.fy
            target_px = self.height * view_ratio

        dist = (2.5 * radius * f) / target_px

        look_at = -self.R[2]

        new_pos = center - dist * look_at

        self.setWorld2Camera(
            center,
            self.R[1],
            new_pos,
        )
        return True

    def toO3DMesh(
        self,
        far: float=1.0,
        color: list=[0, 1, 0],
    ) -> o3d.geometry.LineSet:
        """
        创建相机视锥的可视化网格

        相机坐标系：X右，Y上，Z后（看向 -Z 方向）
        """
        half_width = (self.width / self.fx) * far
        half_height = (self.height / self.fy) * far

        # 在相机坐标系中定义远平面的四个角点
        # 相机看向 -Z 方向，所以远平面在 Z = -far 处
        far_corners_camera = np.array([
            [-half_width, -half_height, -far],  # 左下
            [half_width, -half_height, -far],   # 右下
            [half_width, half_height, -far],    # 右上
            [-half_width, half_height, -far],   # 左上
        ])

        # 将相机坐标系中的点转换到世界坐标系（使用torch计算）
        far_corners_camera_torch = torch.from_numpy(far_corners_camera).to(
            dtype=self.dtype, device=self.device)
        far_corners_camera_homo = torch.cat([
            far_corners_camera_torch, 
            torch.ones((4, 1), dtype=self.dtype, device=self.device)
        ], dim=1)
        far_corners_world_homo = far_corners_camera_homo @ self.camera2world.T
        far_corners_world = toNumpy(far_corners_world_homo[:, :3])

        pos = toNumpy(self.pos)
        vertices = np.vstack([far_corners_world, pos.reshape(1, 3)])

        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # 远平面矩形
            [4, 0], [4, 1], [4, 2], [4, 3],  # 连接相机中心到四个角点
        ], dtype=np.int64)

        frustum = o3d.geometry.LineSet()
        frustum.points = o3d.utility.Vector3dVector(vertices)
        frustum.lines = o3d.utility.Vector2iVector(lines)
        frustum.paint_uniform_color(color)

        return frustum

    def getWorld2NVDiffRast(
        self,
        bbox_length: Union[torch.Tensor, np.ndarray, list]=[2, 2, 2],
    ) -> torch.Tensor:
        """
        构建 nvdiffrast 的 world -> clip(MVP) 变换矩阵，利用 camera 的所有参数（包括外参和内参）

        1. 相机外参：world2camera
        2. 相机内参：fx, fy, cx, cy, width, height
        3. OpenGL 项目经过 Y flip 适配 nvdiffrast
        """

        # 计算合适的 near/far
        bbox_size = torch.linalg.norm(toTensor(bbox_length, self.dtype, self.device))
        near = bbox_size * 0.1
        far = bbox_size * 10.0

        # Step 1: 计算 OpenGL 投影矩阵（右手，左下原点 Y 向上）
        proj = torch.zeros((4, 4), dtype=torch.float32, device=self.device)

        proj[0, 0] = 2 * self.fx / self.width
        proj[1, 1] = 2 * self.fy / self.height
        proj[0, 2] = 1 - 2 * self.cx / self.width
        proj[1, 2] = 2 * self.cy / self.height - 1
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1.0

        # Step 2: OpenGL 到 nvdiffrast: Y 轴翻转，即 NDC 原点从左下变左上。等价于：
        ndc_y_flip = torch.eye(4, dtype=torch.float32, device=self.device)
        ndc_y_flip[1, 1] = -1

        # 总变换矩阵
        mvp = ndc_y_flip @ proj @ self.world2camera

        return mvp

    def toDict(self) -> dict:
        data_dict = {
            'width': self.width,
            'height': self.height,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'world2camera': toNumpy(self.world2camera, np.float64),
        }

        return data_dict

    def save(
        self,
        save_npy_file_path: str,
        overwrite: bool=False,
    ) -> bool:
        if os.path.exists(save_npy_file_path):
            if not overwrite:
                return True

            removeFile(save_npy_file_path)

        data_dict = self.toDict()

        createFileFolder(save_npy_file_path)

        np.save(save_npy_file_path, data_dict, allow_pickle=True)
        return True

    def outputInfo(
        self,
        info_level: int = 0,
    ) -> bool:
        line_start = '\t' * info_level

        # 从旋转矩阵中提取坐标轴
        R = self.R
        x_axis = R[0]  # 相机X轴（右）在世界坐标系中的方向
        y_axis = R[1]  # 相机Y轴（上）在世界坐标系中的方向
        z_axis = R[2]  # 相机Z轴（后）在世界坐标系中的方向

        print(line_start + '[INFO][CameraData]')
        print(line_start + '\t image_size: [', self.width, ',', self.height, ']')
        print(line_start + '\t focal: [', self.fx, ',', self.fy, ',', self.cx, ',', self.cy, ']')
        print(line_start + '\t pos:', toNumpy(self.pos).tolist())
        print(line_start + '\t x_axis (right):', toNumpy(x_axis).tolist())
        print(line_start + '\t y_axis (up):', toNumpy(y_axis).tolist())
        print(line_start + '\t z_axis (back):', toNumpy(z_axis).tolist())
        print(line_start + '\t look_at direction:', (-toNumpy(z_axis)).tolist())
        return True
