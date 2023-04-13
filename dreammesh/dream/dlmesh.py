import os
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import get_encoder

from .obj import Mesh, safe_normalize
from .utils import trunc_rev_sigmoid
from .renderer import Renderer


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class DLMesh(nn.Module):
    def __init__(self, opt, num_layers_bg=2, hidden_dim_bg=16):

        super(DLMesh, self).__init__()

        self.opt = opt

        self.renderer = Renderer()

        # load the template mesh, will calculate normal and texture if not provided.
        self.mesh = Mesh.load_obj(self.opt.mesh, init_empty_tex=self.opt.init_empty_tex)

        # background network
        if not self.opt.skip_bg:
            self.encoder_bg, in_dim_bg = get_encoder('frequency_torch', multires=4)
            self.bg_net = MLP(in_dim_bg, 3, hidden_dim_bg, num_layers_bg)

        # texture parameters
        if not self.opt.lock_tex:
            if self.opt.tex_mlp:
                self.encoder_tex, self.in_dim = get_encoder('hashgrid', interpolation='smoothstep')
                self.tex_net = MLP(self.in_dim, 3, 32, 2)
            else:
                self.raw_albedo = nn.Parameter(trunc_rev_sigmoid(self.mesh.albedo))

        # geo parameters
        if not self.opt.lock_geo:
            if self.opt.geo_mlp:
                self.encoder_geo, in_dim_geo = get_encoder('hashgrid', interpolation='smoothstep')
                self.geo_net = MLP(in_dim_geo, 1, 32, 2)
            else:
                self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v[:, :1]))
        self.name = "mesh"

    def get_params(self, lr):
        params = []

        if not self.opt.skip_bg:
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        if not self.opt.lock_tex:
            if self.opt.tex_mlp:
                params.extend([
                    {'params': self.encoder_tex.parameters(), 'lr': lr * 10},
                    {'params': self.tex_net.parameters(), 'lr': lr},
                ])
            else:
                params.append({'params': self.raw_albedo, 'lr': lr * 10})

        if not self.opt.lock_geo:
            if self.opt.geo_mlp:
                params.extend([
                    {'params': self.encoder_geo.parameters(), 'lr': lr * 10},
                    {'params': self.geo_net.parameters(), 'lr': lr},
                ])
            else:
                params.append({'params': self.v_offsets, 'lr': 0.0001})

        return params

    def get_mesh(self):
        if not self.opt.lock_geo:
            mesh = Mesh(v=self.mesh.v + self.v_offsets * self.mesh.vn, base=self.mesh)
            mesh.auto_normal()
        else:
            mesh = Mesh(base=self.mesh)

        if not self.opt.lock_tex:
            mesh.set_albedo(self.raw_albedo)

        return mesh

    @torch.no_grad()
    def export_mesh(self, save_path):
        if self.opt.lock_tex:
            mesh = self.get_mesh()
            from .obj import save_obj_mesh
            save_obj_mesh(os.path.join(save_path, 'mesh.obj'), mesh.v.cpu().numpy(), mesh.f.cpu().numpy())
        else:
            self.get_mesh().write(os.path.join(save_path, 'mesh.obj'))

    def forward(self, rays_o, rays_d, mvp, h, w, light_d=None, ambient_ratio=1.0, shading='albedo', is_train=True):
        batch = rays_o.shape[0]

        if not self.opt.skip_bg:
            dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            bg_color = torch.sigmoid(self.bg_net(self.encoder_bg(dirs.view(-1, 3)))).view(dirs.shape).contiguous()
        else:
            bg_color = torch.ones(batch, h, w, 3).to(mvp.device)

        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=rays_o.device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        # render
        pr_mesh = self.get_mesh()
        rgb, alpha = self.renderer(pr_mesh, mvp, h, w, light_d, ambient_ratio, shading, self.opt.ssaa,
                                   transform_nml=True if is_train and self.opt.lock_tex else False)

        rgb = rgb * alpha + (1 - alpha) * bg_color

        return {
            "image": rgb,
            "alpha": alpha,
            "pr_mesh": pr_mesh
        }
