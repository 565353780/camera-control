import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. 之前定义的可微模型 ---
class CameraRefiner(nn.Module):
    def __init__(self, rvec_init, tvec_init, K_init):
        super(CameraRefiner, self).__init__()
        # 确保输入是 float32 且为 Tensor
        self.rvec = nn.Parameter(torch.tensor(rvec_init, dtype=torch.float32).flatten())
        self.tvec = nn.Parameter(torch.tensor(tvec_init, dtype=torch.float32).flatten())
        
        # 内参优化变量: fx, fy, cx, cy
        self.intrinsics = nn.Parameter(torch.tensor([
            K_init[0, 0], K_init[1, 1], K_init[0, 2], K_init[1, 2]
        ], dtype=torch.float32))

    def rodrigues_to_matrix(self, rvec):
        theta = torch.norm(rvec)
        if theta < 1e-6:
            return torch.eye(3, device=rvec.device)
        u = rvec / theta
        K_cross = torch.tensor([
            [0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]
        ], device=rvec.device)
        # 罗德里格斯公式: R = I + sin(theta)K + (1-cos(theta))K^2
        R = torch.eye(3, device=rvec.device) + \
            torch.sin(theta) * K_cross + \
            (1 - torch.cos(theta)) * torch.matmul(K_cross, K_cross)
        return R

    def forward(self, points_3d):
        R = self.rodrigues_to_matrix(self.rvec)
        fx, fy, cx, cy = self.intrinsics
        
        # 坐标变换: P_cam = R @ P_3d + t
        points_cam = (R @ points_3d.t()).t() + self.tvec
        
        # 投影 (x/z, y/z)
        # 加上 epsilon 防止除以 0
        eps = 1e-8
        z = points_cam[:, 2:3]
        z = torch.where(z > 0, z, torch.full_like(z, eps)) # 简单处理 Z<0 的点
        
        uv_norm = points_cam[:, :2] / z
        
        # 像素转换
        u = fx * uv_norm[:, 0] + cx
        v = fy * uv_norm[:, 1] + cy
        
        return torch.stack([u, v], dim=1)

# --- 2. 整合后的主逻辑 ---
def solve_and_refine(points, pixels, width, height, fx_init, fy_init):
    # --- STEP 1: 你的 OpenCV 初解逻辑 ---
    K = np.array([
        [fx_init, 0.0, width / 2.0],
        [0.0, fy_init, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.zeros((4, 1))

    # EPnP 初解
    ret, r_init, t_init = cv2.solvePnP(points, pixels, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ret:
        raise RuntimeError("solvePnP EPNP failed")

    # Iterative 精细化 (仅外参)
    ret, r_fine, t_fine = cv2.solvePnP(
        points, pixels, K, dist,
        rvec=r_init, tvec=t_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # --- STEP 2: Torch 联合优化 (内外参) ---
    print("开始 Torch 联合优化...")
    model = CameraRefiner(r_fine, t_fine, K)

    # 使用 LBFGS 或 Adam。对于精细调节，LBFGS 收敛更快更好
    optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)

    # 转换为 Tensor
    pts_3d_t = torch.from_numpy(points).float()
    pts_2d_t = torch.from_numpy(pixels).float()

    def closure():
        optimizer.zero_grad()
        pred = model(pts_3d_t)
        # 重投影误差 Loss
        reproj_loss = torch.mean(torch.sum((pred - pts_2d_t)**2, dim=1))

        # 增加先验约束: 焦距不应偏离初值超过 20% (可选)
        fx_current = model.intrinsics[0]
        reg_loss = 0.01 * torch.pow(fx_current - fx_init, 2)

        loss = reproj_loss + reg_loss
        loss.backward()
        return loss

    # 执行优化步
    for i in range(5): # LBFGS 内部有迭代，这里跑几次大循环即可
        loss = optimizer.step(closure)
        print(f"Step {i}, Loss: {loss.item():.4f}")

    # --- STEP 3: 提取最终结果 ---
    with torch.no_grad():
        final_r = model.rvec.numpy()
        final_t = model.tvec.numpy()
        fx, fy, cx, cy = model.intrinsics.numpy()
        final_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # 构建 4x4 矩阵
        R_mat, _ = cv2.Rodrigues(final_r)

    return R_mat, final_t, final_K
