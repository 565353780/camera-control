import torch
import trimesh
import numpy as np
import nvdiffrast.torch as dr
from typing import Union, Optional

from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera
from flexi_cubes.Module.sh_utils import RGB2SH, SH2RGB, eval_sh  



class NVDiffRastRenderer(object):

    def __init__(self, device: str="cuda") -> None:
        self.glctx = dr.RasterizeCudaContext(device=device) 
        self.device = device 
        

    def _getFaceNormals(
        self,
        verts: torch.Tensor = None,
        faces: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Input mesh or vertices+faces 

        Args:
            mesh (trimesh.Trimesh, optional): trimesh 对象，优先使用
            verts (torch.Tensor, optional): [N, 3] 顶点坐标
            faces (torch.Tensor, optional): [M, 3] 面索引
        Returns:
            face_normals (torch.Tensor): [M, 3, 3] 每个面三个顶点的法向量
        """
        if verts is not None and faces is not None:
            i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
            v0, v1, v2 = verts[i0], verts[i1], verts[i2]
            face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
            face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        else:
            raise ValueError("Please input mesh or vertices+faces")

        face_normals = face_normals[:, None, :].repeat(1, 3, 1) 
        return face_normals 
    

    def _getVertsNormal(
        self,
        verts: torch.Tensor = None,
        faces: torch.Tensor = None
    ) -> torch.Tensor: 
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # [F, 3]
        # faces.reshape(-1) = [f0_v0, f0_v1, f0_v2, f1_v0, ...] 按行交错
        # 因此 face_normals 也需按行交错：每个 face_normal 重复 3 次，与同一面的 3 个顶点对应
        faces_flat = faces.reshape(-1)  # [3F]
        face_normals_rep = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)  # [3F, 3]
        vertex_normals = torch.zeros_like(verts)
        vertex_normals.index_add_(0, faces_flat, face_normals_rep)

        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1, eps=1e-8)
        return vertex_normals
    
    

    def _rasterize(
        self, 
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        vertices_tensor: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Returns:
            tuple: (vertices, faces, vertices_clip, rast_out, rast_out_db )
        """
        trimesh.repair.fix_winding(mesh) 

        if vertices_tensor is not None:
            vertices = vertices_tensor
        else:
            vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device) 

        vertices = vertices.unsqueeze(0) ## shape=[1, V, 3] enable instance mode 
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        
        faces = torch.as_tensor(mesh.faces, dtype=torch.int32, device=self.device)

        mvp = camera.getWorld2NVDiffRast().unsqueeze(0) 
        vertices_clip = torch.bmm(vertices_homo, mvp.transpose(-1, -2)) 


        rast_out, rast_out_db = dr.rasterize(
            self.glctx, vertices_clip, faces,
            resolution=[camera.height, camera.width]
        )

        return vertices, faces, vertices_clip, rast_out, rast_out_db 


    def render(
        self, 
        mesh : trimesh.Trimesh,
        camera: Camera,
        return_types: list = ["mask", "normal", "color", "vertex", "depth"],
        vertices_tensor: Optional[torch.Tensor] = None ,
        shading: bool = False, 
        verts_sh_coeff: Optional[torch.Tensor] = None
    ) -> dict: 
        
        vertices, faces_int, vertices_clip, rast_out, rast_out_db = self._rasterize(
            mesh, camera, vertices_tensor
        )   


        return_dict = {}
        for renderType in return_types: 
            img = None 
            if renderType == "mask": 
                # rast_out: [1, H, W, 4]
                rast_face_idx = rast_out[0, :, :, 3]   # triangle_id + 1, background = 0
                rast_mask = rast_face_idx > 0          # [H, W], bool

                img = dr.antialias(
                    (rast_out[..., -1:] > 0).float(),
                    rast_out,
                    vertices_clip,
                    faces_int
                )

                mask_hw = img[0, :, :, 0]  # [H, W]

                hit_face_ids = torch.unique(rast_face_idx[rast_mask].long() - 1)
                hit_face_ids = hit_face_ids[hit_face_ids >= 0]

                return_dict[renderType] = {
                    "img": img[0],
                    "mask": mask_hw,               # anti-aliased mask, for visualization/loss
                    "rast_mask": rast_mask,        # hard rasterized visibility
                    "rast_face_idx": rast_face_idx,
                    "hit_face_ids": hit_face_ids,
                }
                

            elif renderType == "normal": 
                face_normals = self._getFaceNormals(vertices[0], faces_int ) 
                tri = torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int32).reshape(-1,3) 
                
                img = dr.interpolate(face_normals.reshape(1,-1,3), rast_out, tri)[0]
                img = NVDiffRastRenderer._applyBackground(img, rast_out, [255, 255, 255], camera.device) 
                
                img = dr.antialias(img, rast_out, vertices_clip, faces_int)[0]
                img = (img + 1) / 2 
                img = torch.clamp(img, 0.0, 1.0) 

                return_dict[renderType] = {
                    "img": img 
                } 

            elif renderType == "color": 
                use_texture = False
                texture = None
                uvs = None

                if NVDiffRastRenderer.isTextureExist(mesh):
                    uvs_np = mesh.visual.uv
                    vertices_count = len(mesh.vertices) if vertices_tensor is None else vertices_tensor.shape[0]
                    if uvs_np.shape[0] != vertices_count:
                        print(f'[WARN][NVDiffRastRenderer::renderTexture] UV count ({uvs_np.shape[0]}) != vertex count ({vertices_count}), falling back to vertex colors')
                    else:
                        tex_img = None
                        mat = mesh.visual.material
                        if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                            tex_img = np.array(mat.baseColorTexture)
                        elif hasattr(mat, 'image') and mat.image is not None:
                            tex_img = np.array(mat.image)

                        if tex_img is not None:
                            if len(tex_img.shape) == 2:
                                tex_img = np.stack([tex_img] * 3, axis=-1)
                            elif tex_img.shape[-1] == 4:
                                tex_img = tex_img[:, :, :3]

                            texture = toTensor(tex_img, torch.float32, 'cpu') / 255.0
                            texture = texture.flip(0)
                            uvs = uvs_np
                            use_texture = True

                if not use_texture:
                    image = torch.ones((1, rast_out.shape[1], rast_out.shape[2], 3), dtype=torch.float32, device=camera.device )
                    return_dict[renderType] = {
                        "img": image[0] 
                    } 
                    continue 
                    raise ValueError("No texture!") 
                

                uvs_tensor = toTensor(uvs, torch.float32, camera.device)
                texture = texture.unsqueeze(0).to(camera.device) 

                uv_interp, _ = dr.interpolate(uvs_tensor.unsqueeze(0).contiguous(), rast_out, faces_int) 
                image = dr.texture(texture, uv_interp, filter_mode='linear') 
                image = torch.clamp(image, 0.0, 1.0)

                if shading: 
                    vertex_normals = self._getVertsNormal(vertices[0], faces_int )
                    normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces_int )
                    normals_interp = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)

                    light_direction = [1, 1, 1] 
                    light_dir_cam = toTensor(light_direction, torch.float32, camera.device)
                    light_dir_cam = torch.nn.functional.normalize(light_dir_cam, dim=0, eps=1e-8)

                    # normals_cam = torch.matmul(image, camera.R.T)
                    normals_cam = torch.matmul(normals_interp, camera.R.T) 
                    diffuse = torch.clamp(
                        torch.sum(normals_cam * light_dir_cam, dim=-1), min=0.0, max=1.0
                    )
                    shading = 0.3 + 0.7 * diffuse
                    image = image * shading.unsqueeze(-1)

                image = NVDiffRastRenderer._applyBackground(image, rast_out, [255, 255, 255], camera.device)
                image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces_int) 


                return_dict[renderType] = {
                    "img": image[0] 
                } 

            elif renderType == "vertex" :
                if verts_sh_coeff is not None: 
                    deg = 2 
                    view_dirs = vertices[0] - camera.pos 

                    out_rgb = eval_sh(deg, verts_sh_coeff, view_dirs )
                    out_rgb = SH2RGB(out_rgb)
                    out_rgb = torch.clamp(out_rgb, 0, 1) 
                    
                    # [V,3] → [1,V,3]
                    vertex_color = out_rgb.unsqueeze(0).contiguous()

                    # rasterization interpolation
                    image, _ = dr.interpolate(vertex_color, rast_out, faces_int)

                    # antialias
                    image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces_int)

                    # background
                    image = NVDiffRastRenderer._applyBackground(
                        image, rast_out, [255,255,255], camera.device
                    )

                    return_dict[renderType] = {
                        "img": image[0]
                    }   
                else: 
                    image = torch.ones((1, rast_out.shape[1], rast_out.shape[2], 3), dtype=torch.float32, device=camera.device )
                    return_dict[renderType] = {
                        "img": image[0] 
                    } 


            elif renderType == "depth": 
                vertices_interp, _ = dr.interpolate(vertices, rast_out, faces_int) 

                R_row2 = camera.R[2, :]
                depth = -(torch.matmul(vertices_interp[0], R_row2) + camera.t[2])  # [H, W]

                mask = rast_out[0, :, :, 3] > 0
                depth = torch.where(mask, depth, torch.zeros_like(depth))

                valid_depth = depth[mask]
                if valid_depth.numel() > 0:
                    depth_min = valid_depth.min()
                    depth_range = valid_depth.max() - depth_min
                    depth_normalized = (depth - depth_min) / (depth_range + 1e-8)
                else:
                    depth_normalized = torch.zeros_like(depth)

                depth_normalized = torch.where(mask, depth_normalized, torch.ones_like(depth_normalized))

                # 单通道 antialias 后再扩展到 3 通道
                depth_1ch = depth_normalized.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1] 


                # if enable_antialias:
                depth_1ch = dr.antialias(depth_1ch.contiguous(), rast_out, vertices_clip, faces_int)

                depth_vis = depth_1ch.expand(-1, -1, -1, 3)  # [1, H, W, 3]
                image = NVDiffRastRenderer._applyBackground(depth_vis, rast_out, [255,255,255], camera.device)

                return_dict[renderType] = {
                    "depth": depth, 
                    "img": image[0] 
                } 
                

            else: 
                raise ValueError("no such render type: ", renderType )


        return return_dict 


    @staticmethod
    def _applyBackground(
        image: torch.Tensor,
        rast_out: torch.Tensor,
        bg_color: list,
        device: str,
    ) -> torch.Tensor:
        """将背景色应用到 image [1, H, W, C] 的非物体区域。"""
        mask = rast_out[:, :, :, 3:4] > 0  # [1, H, W, 1]
        bg = toTensor(bg_color[:3], torch.float32, device) / 255.0
        # reshape 到 [1, 1, 1, C] 并利用广播
        bg = bg.view(1, 1, 1, -1)
        return torch.where(mask, image, bg.expand_as(image))

    @staticmethod
    def renderMask(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """
        渲染二值mask图

        Returns:
            dict: mask [H,W], rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertices, faces, _, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor, need_normals=False
        )

        # 单通道 antialias 比 3 通道快 3 倍
        mask_1ch = (rast_out[:, :, :, 3:4] > 0).float()  # [1, H, W, 1]

        if enable_antialias:
            mask_1ch = dr.antialias(mask_1ch.contiguous(), rast_out, vertices_clip, faces)

        mask_hw = mask_1ch[0, :, :, 0]  # [H, W]

        return {
            'mask': mask_hw,
            'rgb': mask_hw.unsqueeze(-1).expand(-1, -1, 3),
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderVertexColor(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        light_direction: Union[torch.Tensor, np.ndarray, list] = [1, 1, 1],
        paint_color: Optional[list] = None,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """
        渲染基于法向的 shading 图

        Returns:
            dict: rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        if paint_color is not None:
            paint_color_tensor = toTensor(paint_color, torch.float32, camera.device)[:3]
            if paint_color_tensor.max() > 1.0:
                paint_color_tensor = paint_color_tensor / 255.0
            vertex_colors = paint_color_tensor.unsqueeze(0).expand(vertices.shape[0], -1)
        elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = toTensor(mesh.visual.vertex_colors[:, :3], torch.float32, camera.device) / 255.0
        else:
            vertex_colors = torch.ones((vertices.shape[0], 3), dtype=torch.float32, device=camera.device)

        normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces)
        normals_interp = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)

        colors_interp, _ = dr.interpolate(vertex_colors.unsqueeze(0).contiguous(), rast_out, faces)
        colors_interp = torch.clamp(colors_interp, 0.0, 1.0)

        light_dir_cam = toTensor(light_direction, torch.float32, camera.device)
        light_dir_cam = torch.nn.functional.normalize(light_dir_cam, dim=0, eps=1e-8)

        normals_cam = torch.matmul(normals_interp, camera.R.T)

        diffuse = torch.clamp(
            torch.sum(normals_cam * light_dir_cam, dim=-1), min=0.0, max=1.0
        )
        shading = 0.3 + 0.7 * diffuse

        image = colors_interp * shading.unsqueeze(-1)

        image = NVDiffRastRenderer._applyBackground(image, rast_out, bg_color, camera.device)

        if enable_antialias:
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)

        return {
            'rgb': image[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        } 

    @staticmethod
    def renderTexture(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        paint_color: Optional[list] = None,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """
        渲染网格纹理颜色（无光照）。无纹理时 fallback 到 renderVertexColor。

        Returns:
            dict: rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        # 先检查纹理再决定是否需要光栅化，避免 fallback 时重复光栅化
        use_texture = False
        texture = None
        uvs = None

        if NVDiffRastRenderer.isTextureExist(mesh):
            uvs_np = mesh.visual.uv
            vertices_count = len(mesh.vertices) if vertices_tensor is None else vertices_tensor.shape[0]
            if uvs_np.shape[0] != vertices_count:
                print(f'[WARN][NVDiffRastRenderer::renderTexture] UV count ({uvs_np.shape[0]}) != vertex count ({vertices_count}), falling back to vertex colors')
            else:
                tex_img = None
                mat = mesh.visual.material
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    tex_img = np.array(mat.baseColorTexture)
                elif hasattr(mat, 'image') and mat.image is not None:
                    tex_img = np.array(mat.image)

                if tex_img is not None:
                    if len(tex_img.shape) == 2:
                        tex_img = np.stack([tex_img] * 3, axis=-1)
                    elif tex_img.shape[-1] == 4:
                        tex_img = tex_img[:, :, :3]

                    texture = toTensor(tex_img, torch.float32, 'cpu') / 255.0
                    texture = texture.flip(0)
                    uvs = uvs_np
                    use_texture = True

        if not use_texture:
            return NVDiffRastRenderer.renderVertexColor(
                mesh=mesh, camera=camera, light_direction=[1, 1, 1],
                paint_color=paint_color, bg_color=bg_color,
                vertices_tensor=vertices_tensor, enable_antialias=enable_antialias,
            )

        # 纹理路径不需要法线
        vertices, faces, _, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor, need_normals=False
        )

        uvs_tensor = toTensor(uvs, torch.float32, camera.device)
        texture = texture.unsqueeze(0).to(camera.device)

        uv_interp, _ = dr.interpolate(uvs_tensor.unsqueeze(0), rast_out, faces)
        image = dr.texture(texture, uv_interp, filter_mode='linear')

        image = NVDiffRastRenderer._applyBackground(image, rast_out, bg_color, camera.device)

        if enable_antialias:
            image = dr.antialias(image.contiguous(), rast_out, vertices_clip, faces)

        return {
            'rgb': image[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderDepth(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """
        渲染深度图

        Returns:
            dict: depth [H,W], rgb [H,W,3], rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertices, faces, _, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor, need_normals=False
        )

        vertices_interp, _ = dr.interpolate(vertices.unsqueeze(0), rast_out, faces)

        # depth = -(vertices_interp[0] @ R.T + t)[:,:,2]
        #       = -(vertices_interp[0] @ R.T[:, 2] + t[2])
        #       = -(vertices_interp[0] @ R[2, :] + t[2])
        R_row2 = camera.R[2, :]
        depth = -(torch.matmul(vertices_interp[0], R_row2) + camera.t[2])  # [H, W]

        mask = rast_out[0, :, :, 3] > 0
        depth = torch.where(mask, depth, torch.zeros_like(depth))

        valid_depth = depth[mask]
        if valid_depth.numel() > 0:
            depth_min = valid_depth.min()
            depth_range = valid_depth.max() - depth_min
            depth_normalized = (depth - depth_min) / (depth_range + 1e-8)
        else:
            depth_normalized = torch.zeros_like(depth)

        depth_normalized = torch.where(mask, depth_normalized, torch.ones_like(depth_normalized))

        # 单通道 antialias 后再扩展到 3 通道
        depth_1ch = depth_normalized.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]

        if enable_antialias:
            depth_1ch = dr.antialias(depth_1ch.contiguous(), rast_out, vertices_clip, faces)

        depth_vis = depth_1ch.expand(-1, -1, -1, 3)  # [1, H, W, 3]
        image = NVDiffRastRenderer._applyBackground(depth_vis, rast_out, bg_color, camera.device)

        return {
            'depth': depth,
            'rgb': image[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderVertexNormal(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """
        渲染顶点法向图（平滑着色），法线在面内通过重心插值平滑过渡。

        Returns:
            dict: world [H,W,3], camera [H,W,3], rgb_world [H,W,3], rgb_camera [H,W,3],
                  rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        vertices, faces, vertex_normals, rast_out, rast_out_db, vertices_clip = NVDiffRastRenderer._rasterize(
            mesh, camera, vertices_tensor
        )

        normals_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast_out, faces)
        normals_world = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)

        normals_camera = torch.matmul(normals_world, camera.R.T)

        normals_world_mapped = normals_world * 0.5 + 0.5
        normals_camera_mapped = normals_camera * 0.5 + 0.5

        mask_4d = rast_out[:, :, :, 3:4] > 0  # [1, H, W, 1]
        bg = toTensor(bg_color[:3], torch.float32, camera.device) / 255.0
        bg = bg.view(1, 1, 1, 3)
        normals_world_vis = torch.where(mask_4d, normals_world_mapped, bg.expand_as(normals_world_mapped))
        normals_camera_vis = torch.where(mask_4d, normals_camera_mapped, bg.expand_as(normals_camera_mapped))

        if enable_antialias:
            combined = torch.cat([normals_world_vis, normals_camera_vis], dim=-1)  # [1, H, W, 6]
            combined = dr.antialias(combined.contiguous(), rast_out, vertices_clip, faces)
            normals_world_vis = combined[:, :, :, :3]
            normals_camera_vis = combined[:, :, :, 3:]

        return {
            'world': normals_world[0],
            'camera': normals_camera[0],
            'rgb_world': normals_world_vis[0],
            'rgb_camera': normals_camera_vis[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderFaceNormal(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """
        渲染面法向图（flat 着色），每个三角面内法线恒定，面与面之间有明显棱角。

        通过将面法线复制到每个顶点并重建索引，使 nvdiffrast 插值后仍保持面内法线一致。

        Returns:
            dict: world [H,W,3], camera [H,W,3], rgb_world [H,W,3], rgb_camera [H,W,3],
                  rasterize_output [H,W,4], bary_derivs [H,W,4]
        """
        if vertices_tensor is not None:
            vertices = vertices_tensor
        else:
            vertices = toTensor(mesh.vertices, torch.float32, camera.device)

        faces = toTensor(mesh.faces, torch.int32, camera.device)

        v0 = vertices[faces[:, 0]]  # [F, 3]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # [F, 3]
        face_normals = torch.nn.functional.normalize(face_normals, dim=1, eps=1e-8)

        # 每个面的法线复制 3 份给该面的 3 个顶点，构造 flat 属性数组 [F*3, 3]
        flat_normals = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        flat_pos_idx = torch.arange(
            faces.shape[0] * 3, device=faces.device, dtype=torch.int32
        ).reshape(-1, 3)

        # 同样需要将顶点坐标按面展开，使 clip 坐标与 flat_pos_idx 对齐
        flat_vertices = torch.cat([v0, v1, v2], dim=1).reshape(-1, 3)  # [F*3, 3]

        vertices_homo = torch.cat([
            flat_vertices,
            torch.ones((flat_vertices.shape[0], 1), dtype=torch.float32, device=camera.device),
        ], dim=1)

        mvp = camera.getWorld2NVDiffRast()
        vertices_clip = torch.matmul(vertices_homo, mvp.T).unsqueeze(0).contiguous()

        print("zz:", vertices_clip.shape )

        glctx = NVDiffRastRenderer.getGlctx(camera.device)
        rast_out, rast_out_db = dr.rasterize(
            glctx, vertices_clip, flat_pos_idx,
            resolution=[camera.height, camera.width],
        )

        normals_interp, _ = dr.interpolate(flat_normals.unsqueeze(0), rast_out, flat_pos_idx)
        normals_world = torch.nn.functional.normalize(normals_interp, dim=-1, eps=1e-8)

        normals_camera = torch.matmul(normals_world, camera.R.T)

        normals_world_mapped = normals_world * 0.5 + 0.5
        normals_camera_mapped = normals_camera * 0.5 + 0.5

        mask_4d = rast_out[:, :, :, 3:4] > 0  # [1, H, W, 1]
        bg = toTensor(bg_color[:3], torch.float32, camera.device) / 255.0
        bg = bg.view(1, 1, 1, 3)
        normals_world_vis = torch.where(mask_4d, normals_world_mapped, bg.expand_as(normals_world_mapped))
        normals_camera_vis = torch.where(mask_4d, normals_camera_mapped, bg.expand_as(normals_camera_mapped))

        if enable_antialias:
            combined = torch.cat([normals_world_vis, normals_camera_vis], dim=-1)  # [1, H, W, 6]
            combined = dr.antialias(combined.contiguous(), rast_out, vertices_clip, flat_pos_idx)
            normals_world_vis = combined[:, :, :, :3]
            normals_camera_vis = combined[:, :, :, 3:]

        return {
            'world': normals_world[0],
            'camera': normals_camera[0],
            'rgb_world': normals_world_vis[0],
            'rgb_camera': normals_camera_vis[0],
            'rasterize_output': rast_out[0],
            'bary_derivs': rast_out_db[0] if rast_out_db is not None else torch.zeros_like(rast_out[0]),
        }

    @staticmethod
    def renderNormal(
        mesh: Union[trimesh.Trimesh, trimesh.Scene],
        camera: Camera,
        bg_color: list = [255, 255, 255],
        vertices_tensor: Optional[torch.Tensor] = None,
        enable_antialias: bool = True,
    ) -> dict:
        """向后兼容别名，等价于 renderVertexNormal。"""
        return NVDiffRastRenderer.renderVertexNormal(
            mesh, camera, bg_color, vertices_tensor, enable_antialias,
        )



    @staticmethod
    def isTextureExist(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> bool:
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None or len(mesh.visual.uv) == 0:
            return False

        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            if hasattr(mesh.visual.material, 'baseColorTexture') and mesh.visual.material.baseColorTexture is not None:
                return True
            if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
                return True

        return False

    