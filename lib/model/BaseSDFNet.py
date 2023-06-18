import os
import sys
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from .net_util import gradient, MetaMLP
from skimage import measure

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from lib.common.sdf import create_grid, eval_grid_octree, eval_grid


class BaseSDFNet(nn.Module):
    def __init__(self, cfg, load_mean_param=False):
        super(BaseSDFNet, self).__init__()
        self.opt = cfg.net
        self.loss_type = self.opt.loss_type
        # self.num_surf = cfg.dataset.num_surface
        # self.num_perturb = cfg.dataset.num_perturb
        # self.num_bbox = cfg.dataset.num_bbox

        self.num_surf = cfg.dataset.num_surface
        self.num_perturb = self.num_surf // 2
        self.num_bbox = self.num_surf // 4

        self.sdf_net = MetaMLP(**cfg.net.sdfnet_kwargs)
        if load_mean_param:
            self.load_mean_shape_param()

    def load_mean_shape_param(self):
        model = torch.load('/media/liaotingting/usb2/projects/ARWild/out/ckpt/igr/model.pt')['model']
        state_dict = OrderedDict()
        for k, v in model.items():
            new_key = k[3:]
            state_dict[new_key] = v
        self.sdf_net.load_state_dict(state_dict)
        return state_dict

    def get_error(self, pred_sdf, pred_normal, gt_normal):
        err_dict = {k: 0 for k in self.loss_type}
        total_error = 0

        if 'igr_sdf' in self.loss_type:
            sdf_surf, _, sdf_bbox = pred_sdf.split([self.num_surf, self.num_perturb, self.num_bbox], dim=1)
            error_surf_sdf = self.opt.lambda_igr_surf_sdf * sdf_surf.abs().mean()
            error_off_surf_sdf = self.opt.lambda_igr_off_sdf * torch.exp(-100.0 * sdf_bbox.abs()).mean()
            err_dict['igr_sdf'] += error_surf_sdf.item() + error_off_surf_sdf.item()
            total_error += error_surf_sdf + error_off_surf_sdf

        if 'normal' in self.loss_type:
            error_nml = (pred_normal[:, :self.num_surf] - gt_normal).norm(2, dim=2).mean()
            error_nml = self.opt.lambda_nml * error_nml
            err_dict['normal'] += error_nml.item()
            total_error += error_nml

        if 'gradient' in self.loss_type:
            error_grad = self.opt.lambda_grad * (pred_normal[:, self.num_surf:].norm(2, dim=2) - 1).pow(2).mean()
            err_dict['gradient'] += error_grad.item()
            total_error += error_grad

        return total_error, err_dict

    @staticmethod
    def derive_normal(points, output, normalize=False):
        grad = gradient(points, output)[:, :, -3:]
        if normalize:
            grad = F.normalize(grad, eps=1e-6, dim=2)
        return grad

    def infer(self, cuda, params=None, resolution=256,
              b_min=np.array([-1, -1, -1]),
              b_max=np.array([1, 1, 1]),
              num_samples=100000, thresh=0, use_octree=True):

        self.sdf_net = self.sdf_net.to(cuda)
        self.sdf_net.eval()

        if params is not None:
            for k, v in params.items():
                params[k] = v.to(cuda)

        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)

        # Then we define the lambda function for cell evaluation
        def eval_func(points):
            samples = np.repeat(np.expand_dims(points, 0), 1, axis=0)
            samples = torch.from_numpy(samples).transpose(1, 2).to(cuda).float()
            pred = self.sdf_net(samples, params) if params is not None else self.sdf_net(samples)
            return pred.squeeze().detach().cpu().numpy()

        # Then we evaluate the grid
        if use_octree:
            sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
        else:
            sdf = eval_grid(coords, eval_func, num_samples=num_samples)

        # if grid_mask is not None:
        #     sdf = sdf * grid_mask + (1-grid_mask) * (thresh + 1.)

        try:
            verts, faces, normals, values = measure.marching_cubes(sdf, thresh)
            verts = np.matmul(verts, mat[:3, :3].T) + mat[:3, 3:4].T
            normals = normals * 0.5 + 0.5
            return verts, faces, normals, values
        except Exception as e:
            print(e)
            return

    def forward_(self, surf_points, latent_input=None, gt_normal=None):
        """
        :param latent_input: [batch, 3, 512, 512]
        :param surf_points:  [1, 18432, 3]
        :param gt_normal: [1, 18432, 3]
        :return:
        """
        if latent_input is not None:
            self.update_sdf_net_param(latent_input)

        surf_points.requires_grad_()

        # get sdf
        res = self.get_res(points, self.sdf_net_params)

        # derive normal
        pred_normal = self.derive_normal(points, res)

        # compute loss
        if gt_normal is not None:
            loss, error_dict = self.get_error(res, pred_normal, gt_normal)
            return res, loss, error_dict

        return res


def get_mean_shape_param():
    model = torch.load('/media/liaotingting/usb2/projects/ARWild/out/ckpt/igr/model.pt')['model']
    state_dict = {}
    for k, v in model.items():
        new_key = 'net.' + k[3:]
        state_dict[new_key] = v
    return state_dict
