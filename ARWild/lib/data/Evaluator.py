# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


import os
import numpy as np
import torch
from torch import nn
import trimesh
import cv2
import os.path as osp
from scipy.spatial import cKDTree
import trimesh.proximity
import trimesh.sample
from PIL import Image
from pytorch3d.ops.knn import knn_points
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lib.data.mesh_util import scalar_to_color, save_obj_mesh_with_color
from lib.common.render import Render


class Evaluator:
    _normal_render = None

    @staticmethod
    def init_gl():
        from lib.renderer.gl.normal_render import NormalRender
        # TODO: replace normal render use pytorch3d
        Evaluator._normal_render = NormalRender(width=512, height=512)

    def __init__(self, device):
        self.device = device
        self.render = Render(device=self.device)
        self.error_term = nn.MSELoss()

        self.offset = 0.0
        self.scale_factor = None

    def set_mesh(self, src_mesh, tgt_mesh, scale_factor=1.0, offset=0.0):
        self.tgt_mesh = tgt_mesh
        self.src_mesh = src_mesh

        # scale mesh to 256

        self.scale_factor = scale_factor
        self.offset = offset

    def _render_normal(self, mesh, deg, norms=None):
        view_mat = np.identity(4)
        rz = deg / 180.0 * np.pi
        model_mat = np.identity(4)
        model_mat[:3, :3] = self._normal_render.euler_to_rot_mat(0, rz, 0)
        model_mat[1, 3] = self.offset
        view_mat[2, 2] *= -1

        self._normal_render.set_matrices(view_mat, model_mat)
        if norms is None:
            norms = mesh.vertex_normals
        self._normal_render.set_normal_mesh(self.scale_factor * mesh.vertices,
                                            mesh.faces, norms, mesh.faces)
        self._normal_render.draw()
        normal_img = self._normal_render.get_color()
        return normal_img

    def render_mesh_list(self, mesh_lst):

        self.offset = 0.0
        self.scale_factor = 1.0

        full_list = []
        for mesh in mesh_lst:
            row_lst = []
            for deg in np.arange(0, 360, 90):
                normal = self._render_normal(mesh, deg)
                row_lst.append(normal)
            full_list.append(np.concatenate(row_lst, axis=1))

        res_array = np.concatenate(full_list, axis=0)

        return res_array

    def _get_reproj_normal_error(self, deg):

        tgt_normal = self._render_normal(self.tgt_mesh, deg)
        src_normal = self._render_normal(self.src_mesh, deg)
        error = (((src_normal[:, :, :3] -
                   tgt_normal[:, :, :3]) ** 2).sum(axis=2).mean(axis=(0, 1)))

        return error, [src_normal, tgt_normal]

    def render_normal(self, verts, faces):

        verts = verts[0].detach().cpu().numpy()
        faces = faces[0].detach().cpu().numpy()

        mesh_F = trimesh.Trimesh(verts * np.array([1.0, -1.0, 1.0]), faces)
        mesh_B = trimesh.Trimesh(verts * np.array([1.0, -1.0, -1.0]), faces)

        self.scale_factor = 1.0

        normal_F = self._render_normal(mesh_F, 0)
        normal_B = self._render_normal(mesh_B,
                                       0,
                                       norms=mesh_B.vertex_normals *
                                             np.array([-1.0, -1.0, 1.0]))

        mask = normal_F[:, :, 3:4]
        normal_F = (torch.as_tensor(2.0 * (normal_F - 0.5) * mask).permute(
            2, 0, 1)[:3, :, :].float().unsqueeze(0).to(self.device))
        normal_B = (torch.as_tensor(2.0 * (normal_B - 0.5) * mask).permute(
            2, 0, 1)[:3, :, :].float().unsqueeze(0).to(self.device))

        return {"T_normal_F": normal_F, "T_normal_B": normal_B}

    def calculate_normal_consist(
            self,
            frontal=True,
            back=True,
            left=True,
            right=True,
            save_demo_img=None,
            return_demo=False,
    ):
        # reproj error
        # if save_demo_img is not None, save a visualization at the given path (etc, "./test.png")
        if self._normal_render is None:
            self.init_gl()
            # print(
            #     "In order to use normal render, "
            #     "you have to call init_gl() before initialing any evaluator objects."
            # )
            # return -1

        side_cnt = 0
        total_error = 0
        demo_list = []

        if frontal:
            side_cnt += 1
            error, normal_lst = self._get_reproj_normal_error(0)
            total_error += error
            demo_list.append(np.concatenate(normal_lst, axis=0))
        if back:
            side_cnt += 1
            error, normal_lst = self._get_reproj_normal_error(180)
            total_error += error
            demo_list.append(np.concatenate(normal_lst, axis=0))
        if left:
            side_cnt += 1
            error, normal_lst = self._get_reproj_normal_error(90)
            total_error += error
            demo_list.append(np.concatenate(normal_lst, axis=0))
        if right:
            side_cnt += 1
            error, normal_lst = self._get_reproj_normal_error(270)
            total_error += error
            demo_list.append(np.concatenate(normal_lst, axis=0))
        if save_demo_img is not None:
            res_array = np.concatenate(demo_list, axis=1)
            res_img = Image.fromarray((res_array * 255).astype(np.uint8))
            res_img.save(save_demo_img)

        if return_demo:
            res_array = np.concatenate(demo_list, axis=1)
            return res_array
        else:
            return total_error

    def export_mesh(self, dir, name):
        self.tgt_mesh.visual.vertex_colors = np.array([255, 0, 0])
        self.src_mesh.visual.vertex_colors = np.array([0, 255, 0])

        (self.tgt_mesh + self.src_mesh).export(
            osp.join(dir, f"{name}_gt_pr.obj"))

    def point_to_surface_dist(self, samples, mesh, factor=100):
        samples = torch.from_numpy(samples).float().to(self.device)
        surface_pts = torch.from_numpy(mesh.vertices).float().to(self.device)
        dist = knn_points(samples[None], surface_pts[None]).dists[0, :, 0].sqrt().cpu().numpy()
        dist[np.isnan(dist)] = 0
        return dist * factor

    def calculate_chamfer_p2s(self, sampled_points=10000):
        """calculate the geometry metrics [chamfer, p2s, chamfer_H, p2s_H]

        Args:
            verts_gt (torch.cuda.tensor): [N, 3]
            faces_gt (torch.cuda.tensor): [M, 3]
            verts_pr (torch.cuda.tensor): [N', 3]
            faces_pr (torch.cuda.tensor): [M', 3]
            sampled_points (int, optional): use smaller number for faster testing. Defaults to 1000.

        Returns:
            tuple: chamfer, p2s, chamfer_H, p2s_H
        """
        gt_surface_pts, _ = trimesh.sample.sample_surface_even(self.tgt_mesh, sampled_points)
        pred_surface_pts, _ = trimesh.sample.sample_surface_even(self.src_mesh, sampled_points)

        dist_pred_gt = self.point_to_surface_dist(gt_surface_pts, self.src_mesh)
        dist_gt_pred = self.point_to_surface_dist(pred_surface_pts, self.tgt_mesh)

        chamfer_dist = 0.5 * (dist_pred_gt.mean() + dist_gt_pred.mean()).item()
        p2s_dist = dist_pred_gt.mean().item()

        return chamfer_dist, p2s_dist

    def normal_consistency_vertex(self):
        """
        :param pred: predicted trimesh
        :param gt trimesh: GT mesh trimesh
        """
        pred_normals = np.array(self.src_mesh.vertex_normals) * 0.5 + 0.5
        gt_normals = np.array(self.tgt_mesh.vertex_normals) * 0.5 + 0.5

        kdtree = cKDTree(self.tgt_mesh.vertices)
        _, ind = kdtree.query(self.src_mesh.vertices)
        gt_normals = gt_normals[ind, :]

        nc_error = ((pred_normals - gt_normals) ** 2).sum(axis=1).mean()

        return nc_error

    def calc_acc(self, output, target, thres=0.5):
        output = output.flatten()
        target = target.flatten()
        # remove the surface points with thres
        non_surf_ids = (target != thres)
        output = output[non_surf_ids]
        target = target[non_surf_ids]

        with torch.no_grad():
            output = output.masked_fill(output < thres, 0.0)
            output = output.masked_fill(output > thres, 1.0)

            target = target.masked_fill(target < thres, 0.0)
            target = target.masked_fill(target > thres, 1.0)

            acc = output.eq(target).float().mean()

            # iou, precison, recall
            output = output > thres
            target = target > thres

            union = output | target
            inter = output & target

            _max = torch.tensor(1.0).to(output.device)

            union = max(union.sum().float(), _max)
            true_pos = max(inter.sum().float(), _max)
            vol_pred = max(output.sum().float(), _max)
            vol_gt = max(target.sum().float(), _max)

            iou = true_pos / union
            prec = true_pos / vol_pred
            recall = true_pos / vol_gt

            return {
                "acc": acc.item(),
                "iou": iou.item(),
                "prec": prec.item(),
                "recall": recall.item()
            }

    def get_error_map(self, save_path=None):
        dists = self.point_to_surface_dist(self.src_mesh.vertices, self.tgt_mesh)
        error_color = scalar_to_color(dists, min=0, max=10)
        self.render.load_mesh(self.src_mesh.vertices, self.src_mesh.faces, error_color, normalize=True)
        error_map = self.render.get_image([0])[..., ::-1]
        if save_path is not None:
            cv2.imwrite(save_path, error_map * 255)
        return error_map

    def get_metrics(self):
        nc = self.normal_consistency_vertex()
        chamfer, p2s = self.calculate_chamfer_p2s()
        return {
            'nc': nc,
            'chamfer': chamfer,
            'p2s': p2s
        }
