"""PBR + image-based-lighting (HDR equirect) shading for nvdiffrast renders.

Cook-Torrance GGX + IBL: SH9 diffuse irradiance + roughness-mip specular env sampling
+ Karis analytic env-BRDF, ACES tonemap -> sRGB. Material follows the UE-style random
(BaseColor = white*U(0.1,0.5), Roughness U(0.4,0.8), Metallic U(0,0.4)).

Usage (precompute env once per scene, material once per object, then per camera):
    ibl = prepare_ibl(hdr_path, device)
    mat = sample_material(seed, device)
    rgb = render_pbr_ibl(mesh, camera, ibl, mat, rasterize_dict=rd)   # [H,W,3] in [0,1]
"""
import os
import glob
import math
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

# 渲染器所属仓库自带的 IBL HDR 资产支持的扩展名（equirect HDR / EXR）。
IBL_HDR_EXTENSIONS = ('.hdr', '.exr')


def getIblDataDir() -> str:
    """内置 IBL HDR 资产目录（随 camera-control 仓库分发）。

    资产放在 ``camera_control/Data/ibl/``（渲染器所属仓库自带渲染资产）。库自身
    通过 :func:`resolveIblHdrPath` 自动定位绝对路径并加载，调用方无需关心其位置。
    路径基于本文件 ``__file__`` 推导，因此无论仓库被放在哪里、被谁导入都成立。
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'ibl')


def listIblHdrFiles(ibl_dir: str = None) -> list:
    """列出目录下所有可用的 HDR/EXR 资产文件（绝对路径，已排序）。

    *ibl_dir* 为 ``None`` 时使用内置资产目录 :func:`getIblDataDir`。
    """
    if ibl_dir is None:
        ibl_dir = getIblDataDir()
    files = []
    for ext in IBL_HDR_EXTENSIONS:
        files += glob.glob(os.path.join(ibl_dir, '*' + ext))
    return sorted(os.path.abspath(p) for p in files)


def resolveIblHdrPath(ibl_hdr_path=None, seed: int = None) -> str:
    """把任意 IBL 入参解析成一个具体、可加载的 HDR 文件绝对路径。

    这是渲染 PBR 灰模时定位环境贴图的单一原子入口，自包含、无外部依赖：

      - ``None``        -> fallback 到内置资产目录（camera-control 自带），从中选一张。
      - 目录            -> 从目录内的 ``.hdr/.exr`` 中选一张（随机环境增强）。
      - 文件            -> 直接使用（校验存在性）。

    目录场景下默认随机挑选（环境光增强）；传入 *seed* 可得到可复现的选择。

    Returns:
        str: HDR 文件的绝对路径。

    Raises:
        FileNotFoundError: 解析后找不到任何可用的 HDR/EXR 文件。
    """
    if not ibl_hdr_path:
        ibl_hdr_path = getIblDataDir()

    if os.path.isdir(ibl_hdr_path):
        files = listIblHdrFiles(ibl_hdr_path)
        if not files:
            raise FileNotFoundError(
                f"no {'/'.join(IBL_HDR_EXTENSIONS)} IBL asset found under {ibl_hdr_path!r}")
        rng = random.Random(seed) if seed is not None else random
        return rng.choice(files)

    path = os.path.abspath(ibl_hdr_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"IBL HDR file not found: {ibl_hdr_path!r}")
    return path


def load_env(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(img[..., ::-1].astype(np.float32))  # BGR(HDR) -> RGB


def _dir_to_uv(d):
    u = 0.5 + torch.atan2(d[..., 0], -d[..., 2]) / (2 * math.pi)
    v = 0.5 - torch.asin(torch.clamp(d[..., 1], -1, 1)) / math.pi
    return torch.stack([u, v], -1)


def _sample(env_t, d):
    grid = (_dir_to_uv(d) * 2 - 1).unsqueeze(0)
    return F.grid_sample(env_t, grid, mode='bilinear', align_corners=False,
                         padding_mode='border')[0].permute(1, 2, 0)


def _project_sh9(env_rgb):
    env = cv2.resize(env_rgb, (128, 64), interpolation=cv2.INTER_AREA).astype(np.float64)
    He, We = env.shape[:2]
    js, is_ = np.mgrid[0:He, 0:We]
    u = (is_ + 0.5) / We
    v = (js + 0.5) / He
    a = (0.5 - v) * math.pi
    y = np.sin(a); c = np.cos(a); phi = (u - 0.5) * 2 * math.pi
    x = c * np.sin(phi); z = -c * np.cos(phi)
    dw = (2 * math.pi / We) * (math.pi / He) * np.cos(a).clip(0)
    Y = [0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x,
         1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
         1.092548 * x * z, 0.546274 * (x * x - y * y)]
    return np.stack([(env * (Y[k] * dw)[..., None]).reshape(-1, 3).sum(0) for k in range(9)])


def _sh_irradiance(N, L):
    x, y, z = N[..., 0], N[..., 1], N[..., 2]
    A = [math.pi, 2 * math.pi / 3, 2 * math.pi / 3, 2 * math.pi / 3,
         math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4]
    Y = [0.282095 + 0 * x, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y,
         1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)]
    E = torch.zeros_like(N)
    for k in range(9):
        E = E + A[k] * Y[k].unsqueeze(-1) * L[k].view(1, 1, 3)
    return E.clamp(min=0)


def _env_brdf_approx(rough, NoV, device):
    c0 = torch.tensor([-1, -0.0275, -0.572, 0.022], device=device)
    c1 = torch.tensor([1, 0.0425, 1.04, -0.04], device=device)
    r = rough.unsqueeze(-1) * c0 + c1
    a004 = torch.minimum(r[..., 0] ** 2, torch.exp2(-9.28 * NoV)) * r[..., 0] + r[..., 1]
    return -1.04 * a004 + r[..., 2], 1.04 * a004 + r[..., 3]


def _aces(x):
    return torch.clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def prepare_ibl(hdr_path, device, spec_levels=6):
    env_rgb = load_env(hdr_path)
    env_t = torch.from_numpy(env_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
    L_sh = torch.from_numpy(_project_sh9(env_rgb)).float().to(device)
    cur = cv2.resize(env_rgb, (512, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    mips = [cur]
    for i in range(1, spec_levels):
        k = 2 * i + 1
        mips.append(cv2.GaussianBlur(mips[-1], (k * 2 + 1, k * 2 + 1), 0))
    spec_mips = [torch.from_numpy(np.ascontiguousarray(m[..., :3])).float()
                 .permute(2, 0, 1).unsqueeze(0).to(device) for m in mips]
    return {'env_t': env_t, 'L_sh': L_sh, 'spec_mips': spec_mips, 'n_lvl': len(spec_mips),
            'hdr_path': hdr_path}


def sample_material(seed, device):
    rng = np.random.default_rng(seed)
    g = float(rng.uniform(0.1, 0.5))
    base = torch.tensor([g, g, g], device=device)
    rough = torch.tensor(float(rng.uniform(0.4, 0.8)), device=device)
    metal = torch.tensor(float(rng.uniform(0.0, 0.4)), device=device)
    F0 = 0.04 * (1 - metal) + base * metal
    return {'base_color': base, 'roughness': rough, 'metallic': metal, 'F0': F0,
            'base_gray': g, 'roughness_f': float(rough), 'metallic_f': float(metal)}


@torch.no_grad()
def shade_pbr_ibl(normal_world, view_dir, ibl, material, exposure=1.2):
    """Atomic per-pixel Cook-Torrance GGX + IBL shader (geometry-source agnostic).

    纯逐像素着色原子函数：不关心几何缓冲来自哪里（mesh 光栅化插值、GS splatting
    累积法线、深度图反投影均可），只要给出世界空间单位法线与单位视线方向即可，
    因此 mesh 灰模与 GS 灰模可共享同一套着色数学，保证风格一致。

    Args:
        normal_world: [H,W,3] unit world-space normals.
        view_dir:     [H,W,3] unit world-space view dirs (surface -> camera).
        ibl:      :func:`prepare_ibl` 的返回。
        material: :func:`sample_material` 的返回。
        exposure: 曝光系数。

    Returns:
        [H,W,3] sRGB in [0,1]。不做背景合成——由调用方按自身的 mask/alpha 合成。
    """
    dev = normal_world.device
    Nw, Vd = normal_world, view_dir
    NoV = torch.clamp((Nw * Vd).sum(-1, keepdim=True), 1e-4, 1.0)
    R = F.normalize(2 * (Nw * Vd).sum(-1, keepdim=True) * Nw - Vd, dim=-1)

    base = material['base_color']; rough = material['roughness']
    metal = material['metallic']; F0 = material['F0']

    irr = _sh_irradiance(Nw, ibl['L_sh'])
    diffuse = (1 - metal) * base.view(1, 1, 3) * irr / math.pi

    n_lvl = ibl['n_lvl']
    lvl = float(rough) * (n_lvl - 1)
    lo = int(math.floor(lvl)); hi = min(lo + 1, n_lvl - 1); fr = lvl - lo
    prefilt = _sample(ibl['spec_mips'][lo], R) * (1 - fr) + _sample(ibl['spec_mips'][hi], R) * fr
    sc, bs = _env_brdf_approx(rough * torch.ones_like(NoV[..., 0]), NoV[..., 0], dev)
    specular = prefilt * (F0.view(1, 1, 3) * sc.unsqueeze(-1) + bs.unsqueeze(-1))

    color = (diffuse + specular) * exposure
    return torch.clamp(_aces(color), 0, 1) ** (1 / 2.2)         # [H,W,3] in [0,1], sRGB


@torch.no_grad()
def render_pbr_ibl(mesh, camera, ibl, material, exposure=1.2,
                   rasterize_dict=None, bg_color=(255, 255, 255), vertices_tensor=None):
    """Render one view with PBR+IBL. Returns rgb [H,W,3] float in [0,1] (object on bg_color).

    Mesh 路径包装器：光栅化得到法线/位置/掩码缓冲，再调用原子着色器
    :func:`shade_pbr_ibl` 完成逐像素着色并合成背景。
    """
    from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer
    from camera_control.Method.data import toTensor

    dev = camera.device
    if rasterize_dict is None:
        rasterize_dict = NVDiffRastRenderer.rasterize(mesh, camera, vertices_tensor)
    rast = rasterize_dict['rast_out']
    faces = rasterize_dict['faces']
    verts = rasterize_dict['vertices']

    mn = np.asarray(mesh.vertex_normals)
    if mn.shape[0] == verts.shape[0]:
        vn = toTensor(mn, torch.float32, dev)
    else:
        vn = NVDiffRastRenderer._computeVertexNormals(verts, faces)

    Nw = F.normalize(dr.interpolate(vn.unsqueeze(0), rast, faces)[0][0], dim=-1)
    Pw = dr.interpolate(verts.unsqueeze(0).contiguous(), rast, faces)[0][0]
    mask = (rast[..., 3:4] > 0).float()[0]

    cam_pos = camera.pos.to(dev).float().view(1, 1, 3)
    Vd = F.normalize(cam_pos - Pw, dim=-1)

    ldr = shade_pbr_ibl(Nw, Vd, ibl, material, exposure=exposure)

    bg = toTensor(list(bg_color), torch.float32, dev).view(1, 1, 3) / 255.0
    rgb = torch.where(mask > 0, ldr, bg)
    return rgb
