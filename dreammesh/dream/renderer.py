import torch
import numpy as np
from dream.obj import Mesh
import torch.nn.functional as F
import nvdiffrast.torch as dr
from . import utils
from .obj import compute_normal


class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.glctx = dr.RasterizeGLContext()

    def get_shading_normal(self):
        # # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
        # v0 = mesh.v[mesh.f[:, 0].long(), :]
        # v1 = mesh.v[mesh.f[:, 1].long(), :]
        # v2 = mesh.v[mesh.f[:, 2].long(), :]
        # face_normals = utils.safe_normalize(torch.cross(v1 - v0, v2 - v0))
        # face_normal_indices = (
        #     torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1,
        #                                                                                               3)
        # gb_geometric_normal, _ = dr.interpolate(face_normals[None, ...], rast, face_normal_indices.int())

        # gb_normal = prepare_shading_normal(alpha, v_clip, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal,
        #                                    two_sided_shading=True, opengl=True)

        # tangent, _ = dr.interpolate(mesh.v_tng[None, ...], rast, mesh.f_tng)  # Interpolate tangents
        pass

    def forward(self, mesh, mvp,
                h=512,
                w=512,
                light_d=None,
                ambient_ratio=1.,
                shading='albedo',
                spp=1,
                return_normal=False,
                transform_nml=False):
        """
        Args:
            spp:
            return_normal:
            transform_nml:
            mesh: Mesh object
            mvp: [batch, 4, 4]
            h: int
            w: int
            light_d:
            ambient_ratio: float
            shading: str shading type albedo, normal,
            ssp: int
        Returns:
            color: [batch, h, w, 3]
            alpha: [batch, h, w, 1]
            depth: [batch, h, w, 1]

        """
        B = mvp.shape[0]
        v_clip = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1),
                              torch.transpose(mvp, 1, 2)).float()  # [B, N, 4]

        res = (int(h * spp), int(w * spp)) if spp > 1 else (h, w)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, mesh.f, res)

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        alpha, _ = dr.interpolate(torch.ones_like(v_clip[..., :1]), rast, mesh.f)  # [B, H, W, 1]
        depth = rast[..., [2]]  # [B, H, W]

        # Compute normal space
        vn = compute_normal(v_clip[0, :, :3], mesh.f) if transform_nml else mesh.vn
        normal, _ = dr.interpolate(vn[None, ...], rast, mesh.fn)

        # Texture coordinate
        if not shading == 'normal':
            texc, texc_db = dr.interpolate(mesh.vt[None, ...], rast, mesh.ft, rast_db=rast_db, diff_attrs='all')

            albedo = dr.texture(
                mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [B, H, W, 3]
            albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device))  # remove background

        # albedo_jitter = dr.texture(
        #     mesh.albedo.unsqueeze(0),
        #     texc + torch.normal(mean=0, std=0.005, size=texc.shape, device="cuda"),
        #     uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [B, H, W, 3]
        # albedo_grad = torch.sum(torch.abs(albedo_jitter[..., 0:3] - albedo[..., 0:3]), dim=-1, keepdim=True) / 3

        ################################################################################
        # Shade
        ################################################################################
        if shading == 'albedo':
            color = albedo
        elif shading == 'normal':
            color = (normal + 1) / 2.
            # color = color[..., [2, 2, 2]]
        else:
            lambertian = ambient_ratio + (1 - ambient_ratio) * (normal @ light_d.view(-1, 1)).float().clamp(min=0)
            color = albedo * lambertian.repeat(1, 1, 1, 3)

        color = dr.antialias(color, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, mesh.f).clamp(0, 1)  # [H, W, 3]

        # inverse super-sampling
        if spp > 1:
            color = utils.scale_img_nhwc(color, (h, w))
            alpha = utils.scale_img_nhwc(alpha, (h, w))
            depth = utils.scale_img_nhwc(depth, (h, w))
            normal = utils.scale_img_nhwc(normal, (h, w))

        if return_normal:
            return color, alpha, (normal + 1) / 2.
        return color, alpha
