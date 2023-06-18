import os
import cv2
import random
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
import smpl
from PIL import Image
from lib.common.geometry import index
from lib.data.mesh_util import compute_normal, get_visibility
from lib.common.train_util import get_mgrid
import torchvision.transforms as transforms

from pytorch3d.ops import knn_points, knn_gather

log = logging.getLogger('trimesh')
log.setLevel(40)


class TestDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.opt = cfg.dataset
        self.data_dir = self.opt.data_dir

        self.image_size = self.opt.image_size
        self.device = torch.device(f"cuda:{cfg.training.gpus[0]}")
        # PIL to tensor
        self.image_to_tensor = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # PIL to tensor
        self.mask_to_tensor = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,))
        ])
        # self.subjects = self.get_subjects()
        self.smpl_model = smpl.create(self.cfg.smpl.path, **{'gender': 'NEUTRAL'})
        self.faces = self.smpl_model.faces.astype(np.int32)

        self.data_name = self.data_dir.split('/')[-1]
        self.im_names = sorted(os.listdir(os.path.join(self.data_dir, 'image')))[:200]

        self.subjects = sorted(Path('./data/splits/rp.txt').read_text().strip().split('\n'))[50:]

    def __len__(self):
        # return len(self.im_names)
        return len(self.subjects)

    def load_smpl(self, smpl_file):
        smpl_data = np.load(smpl_file)
        betas = torch.from_numpy(smpl_data['betas']).view(1, -1).float() if 'betas' in smpl_data else torch.zeros(1, 10)
        poses = torch.from_numpy(smpl_data['pose']).view(1, 24, 3).float()
        calib = smpl_data['calib'].reshape((4, 4)) if 'calib' in smpl_data else self.get_calib(smpl_data)

        with torch.no_grad():
            posed_output = self.smpl_model(betas=betas,
                                           global_orient=poses[:, :1],
                                           body_pose=poses[:, 1:],
                                           custom_out=True)
            smpl_v = posed_output.vertices[0].numpy()
            smpl_v = np.matmul(smpl_v, calib[:3, :3].T) + calib[None, :3, 3]
            smpl_v[:, 1] *= -1
            jT = posed_output.joint_transform[0, :24]
            smpl_mesh = trimesh.Trimesh(smpl_v, self.smpl_model.faces, **{'process': False, 'maintain_order': True})

        return smpl_mesh, jT, calib

    def get_calib(self, param):
        # param = np.load(param_path, allow_pickle=True)
        # pixel unit / world unit
        ortho_ratio = param['ortho_ratio']
        # world unit / model unit
        # replace the scale and center of normalizing canonical mesh with posed mesh
        scale = param['scale']
        # camera center world coordinate
        center = param['center']
        # model rotation
        R = param['R']

        translate = -np.matmul(R, center)
        extrinsic = np.concatenate([R, translate.reshape(3, 1)], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

        # Match camera space to image pixel space
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = scale / ortho_ratio
        scale_intrinsic[1, 1] = -scale / ortho_ratio
        scale_intrinsic[2, 2] = scale / ortho_ratio

        # Match image pixel space to image uv space
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(self.cfg.dataset.image_size // 2)
        uv_intrinsic[1, 1] = 1.0 / float(self.cfg.dataset.image_size // 2)
        uv_intrinsic[2, 2] = 1.0 / float(self.cfg.dataset.image_size // 2)
        # Transform under image pixel space
        trans_intrinsic = np.identity(4)

        # Note that here we replace the origin extrinsic calib in posed space by canonical space
        intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
        calib = np.matmul(intrinsic, extrinsic)

        return calib

    def prepare_data(self, id):
        # sid = self.im_names[id][:-4]
        # param_path = f'{self.data_dir}/smpl/{sid}.npz'
        # icon_f_nml_path = f'{self.data_dir}/normalF/{sid}.png'
        # icon_b_nml_path = f'{self.data_dir}/normalB/{sid}.png'
        # image_path = f'{self.data_dir}/image/{sid}.png'
        # obj_path = f'{self.data_dir}/results/exp3-rp/{sid}_posed.obj'

        # sid = "rp_tenzin_rigged_002_MAX"
        sid = self.subjects[id]
        data_dir = "/media/liaotingting/usb3/Dataset/render_people64/synthetic"
        image_path = f"{data_dir}/{sid}/000/RENDER/000.png"
        mask_path = f"{data_dir}/{sid}/000/MASK/000.png"
        param_path = f"{data_dir}/{sid}/000/PARAM/000.npz"
        icon_f_nml_path = f"{data_dir}/{sid}/000/NORMAL/000.png"
        icon_b_nml_path = f"{data_dir}/{sid}/000/NORMAL/000.png"
        # obj_path = f"./out/res/car-normal-1view/rp/{sid}_posed.obj"
        obj_path = f"../PIFuSOTA/results/rp/pifuhd/{sid}_000_000.obj"

        mask = Image.open(image_path).convert('RGBA').split()[-1]
        mask = self.mask_to_tensor(mask)

        # load normal image and calibration
        nmlF = Image.open(icon_f_nml_path).convert('RGB')
        nmlF = self.image_to_tensor(nmlF)
        nmlF *= mask.expand_as(nmlF)

        nmlB = Image.open(icon_b_nml_path).convert('RGB')
        nmlB = self.image_to_tensor(nmlB)
        nmlB *= mask.expand_as(nmlB)

        smpl_mesh, jT, calib = self.load_smpl(param_path)
        smpl_data = torch.as_tensor(np.concatenate([smpl_mesh.vertices,
                                                    smpl_mesh.vertex_normals], 1)).float()

        mesh = trimesh.load(obj_path)
        # faces = np.concatenate([mesh.faces, smpl_mesh.faces + mesh.faces.max() + 1])
        # vertices = np.concatenate([mesh.vertices, smpl_mesh.vertices])
        # mesh = trimesh.Trimesh(vertices, faces)

        sampler = Sampler(mesh, nmlF, mask, None, num_surface=self.cfg.dataset.num_surface, device=self.device)

        return smpl_mesh, mesh, smpl_data, sampler, sid

    def _prepare_data(self):
        pass

class Sampler:
    def __init__(self, mesh, normalF, mask, normalB=None,
                 num_surface=20000, local_sigma=0.01,
                 device=None):
        self.device = device
        self.normalF = normalF
        self.normalB = normalB
        self.mask = mask.squeeze() > 0.5
        self.no_bg_xy = self.get_no_bg_xy()
        self.set_mesh(mesh)
        self.num_surface = num_surface
        self.local_sigma = local_sigma
        self.n_vis_sample = self.num_surface // 2
        self.n_invis_sample = self.num_surface - self.n_vis_sample
        self.b_min, self.b_max = self.get_bbox(mesh.vertices)
        # self.no_bg_z = torch.nn.Parameter(torch.stack([self.fv_pts[:, 2], self.bv_pts[:, 2]], 1), requires_grad=True)

    def get_no_bg_xy(self):
        h, w = self.mask.shape
        tensors = [torch.linspace(-1., 1., steps=w), torch.linspace(-1., 1., steps=h)]
        grid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        xy = grid[self.mask.bool()][:, [1, 0]] * torch.as_tensor([1., -1.])

        xy = torch.stack(random.choices(xy, k=min(20000, len(xy))))
        return xy

    def sample_no_bg_points(self, points):
        # compute depth
        vis_xy, vis_z = torch.split(points, [2, 1], dim=1)
        _, ids, _ = knn_points(self.no_bg_xy[None], vis_xy[None, :, :2])
        z = knn_gather(vis_z[None], ids)[0, :, 0, :]
        xyz = torch.cat([self.no_bg_xy, z], 1)
        return xyz

    @staticmethod
    def index_image(pts, image):
        """
        :param pts: () [B, 3]
        :param image: [dim, H, W]
        :return:
        """
        xy = pts[:, :2] * torch.as_tensor([1., -1.])
        values = index(image[None], xy[None].transpose(1, 2))
        return values[0].transpose(1, 0)

    def get_bbox(self, vertices):
        h, w = self.mask.shape
        rows = torch.any(self.mask, dim=1)
        cols = torch.any(self.mask, dim=0)
        y_max, y_min = h - torch.where(rows)[0][[0, -1]]
        x_min, x_max = torch.where(cols)[0][[0, -1]]

        x_min = max(x_min.div(w * 0.5) - 1.1, -1.)
        x_max = min(x_max.div(w * 0.5) - 0.9, 1.)
        y_min = max(y_min.div(h * 0.5) - 1.1, -1.)
        y_max = min(y_max.div(h * 0.5) - 0.9, 1.)
        z_min = max(vertices[:, 2].min() - 0.2, -1.)
        z_max = min(vertices[:, 2].max() + 0.2, 1.)

        b_min = torch.as_tensor([x_min, y_min, z_min])
        b_max = torch.as_tensor([x_max, y_max, z_max])
        return b_min, b_max

    def update_normal(self):
        if self.normalF is not None:
            self.fv_nml = self.index_image(self.fv_pts, self.normalF)

        if self.normalB is not None:
            self.bv_nml = self.index_image(self.bv_pts, self.normalB)

    def set_mesh(self, mesh, num_sample=20000, update_nml=True, no_bg_pts=True):
        # update bbox
        self.b_min, self.b_max = self.get_bbox(mesh.vertices)

        # get
        surf_pts = torch.as_tensor(mesh.vertices).float()
        surf_nml = torch.as_tensor(compute_normal(mesh.vertices, mesh.faces)).float()
        vis_ids = get_visibility(mesh.vertices, mesh.faces, self.device)

        fv_ids = vis_ids
        bv_ids = torch.logical_not(vis_ids)

        # # bg_mask = self.index_image(surf_pts, self.mask[None].float())[:, 0].bool()
        if len(vis_ids) > num_sample:
            rand_mask = torch.zeros_like(vis_ids)
            rand_mask[torch.randint(0, len(vis_ids), (num_sample,))] = 1
            rand_mask = rand_mask.bool()
            fv_ids = torch.logical_and(fv_ids, rand_mask)
            bv_ids = torch.logical_and(bv_ids, rand_mask)

        self.fv_pts, self.bv_pts = surf_pts[fv_ids], surf_pts[bv_ids]
        self.fv_nml, self.bv_nml = surf_nml[fv_ids], surf_nml[bv_ids]

        if update_nml:
            if no_bg_pts:
                self.update_normal()
                return
            if self.normalF is not None:
                self.fv_pts = torch.cat([self.fv_pts, self.sample_no_bg_points(self.fv_pts)])
            if self.normalB is not None:
                self.bv_pts = torch.cat([self.bv_pts, self.sample_no_bg_points(self.bv_pts)])
            self.update_normal()

        # save_obj_mesh('back.obj', self.fv_pts.numpy())
        # mesh.export('mesh.obj')
        # print('s')
        # exit()

    def get_points(self):
        ids = torch.as_tensor(np.random.randint(0, len(self.fv_pts), self.n_vis_sample))
        vis_samples = self.fv_pts[ids]
        vis_normals = self.fv_nml[ids]

        ids = torch.as_tensor(np.random.randint(0, len(self.bv_pts), self.n_invis_sample))
        invis_samples = self.bv_pts[ids]
        invis_normals = self.bv_nml[ids]

        surf_samples = torch.cat([vis_samples, invis_samples])
        surf_normals = torch.cat([vis_normals, invis_normals])

        sample_local = torch.stack(random.choices(surf_samples, k=self.num_surface // 2))
        sample_local += torch.randn_like(sample_local) * self.local_sigma

        sample_global = torch.rand(self.num_surface // 4, 3) * (self.b_max - self.b_min) + self.b_min

        sample = torch.cat([surf_samples, sample_local, sample_global])

        # save_obj_mesh('smpl.obj', sample.numpy())
        # exit()
        return sample.float(), surf_normals.float()






