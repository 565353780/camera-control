import torch
import numpy as np
import open3d as o3d
from typing import Union
from copy import deepcopy

from camera_control.Method.data import toTensor


class CameraData(object):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 320.0,
        cy: float = 240.0,
        pos: Union[torch.Tensor, np.ndarray, list] = [0, 0, 0],
        look_at: Union[torch.Tensor, np.ndarray, list] = [1, 0, 0],
        up: Union[torch.Tensor, np.ndarray, list] = [0, 0, 1],
        world2camera: Union[torch.Tensor, np.ndarray, list, None] = None,
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
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        if world2camera is not None:
            self.world2camera = toTensor(world2camera)
        else:
            pos = toTensor(pos)
            self.setWorld2Camera(look_at, up, pos)
        return

    def clone(self):
        return deepcopy(self)

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

        camera2world = torch.eye(4, dtype=self.world2camera.dtype, device=self.world2camera.device)
        camera2world[:3, :3] = R_T  # R的转置
        camera2world[:3, 3] = -R_T @ self.t  # -R^T @ t

        return camera2world

    def setWorld2CameraByRt(
        self,
        R: Union[torch.Tensor, np.ndarray, list],
        t: Union[torch.Tensor, np.ndarray, list],
    ) -> bool:
        R = toTensor(R).reshape(3, 3)
        t = toTensor(t).reshape(3)

        self.world2camera = torch.eye(4, dtype=R.dtype, device=R.device)
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
        look_at = toTensor(look_at)
        up = toTensor(up)

        if pos is None:
            pos = self.pos
        else:
            pos = toTensor(pos)

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

        # 转换到世界坐标系
        camera2world = self.camera2world

        # 将相机坐标系中的点转换到世界坐标系（使用torch计算）
        far_corners_camera_torch = torch.from_numpy(far_corners_camera).to(camera2world.dtype)
        far_corners_camera_homo = torch.cat([
            far_corners_camera_torch, 
            torch.ones((4, 1), dtype=camera2world.dtype)
        ], dim=1)
        far_corners_world_homo = (camera2world @ far_corners_camera_homo.T).T
        far_corners_world = far_corners_world_homo[:, :3].numpy()

        pos = self.pos.numpy()
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

    def outputInfo(
        self,
        info_level: int = 0,
    ) -> bool:
        line_start = '\t' * info_level

        # 从旋转矩阵中提取坐标轴
        R = self.rot
        x_axis = R[0, :]  # 相机X轴（右）在世界坐标系中的方向
        y_axis = R[1, :]  # 相机Y轴（上）在世界坐标系中的方向
        z_axis = R[2, :]  # 相机Z轴（后）在世界坐标系中的方向

        print(line_start + '[INFO][CameraData]')
        print(line_start + '\t image_size: [', self.width, ',', self.height, ']')
        print(line_start + '\t focal: [', self.fx, ',', self.fy, ',', self.cx, ',', self.cy, ']')
        print(line_start + '\t pos:', self.pos.numpy().tolist())
        print(line_start + '\t x_axis (right):', x_axis.numpy().tolist())
        print(line_start + '\t y_axis (up):', y_axis.numpy().tolist())
        print(line_start + '\t z_axis (back):', z_axis.numpy().tolist())
        print(line_start + '\t look_at direction:', (-z_axis).numpy().tolist())
        return True
