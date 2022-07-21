import glob
import os
import random
import logging
from pathlib import Path
from termcolor import colored
import trimesh
from PIL import ImageOps
from PIL.ImageFilter import GaussianBlur
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from .mesh_util import *
from PIL import Image
from glob import glob
import smpl
from lib.common.train_util import warp_and_project_points

log = logging.getLogger('trimesh')
log.setLevel(40)


class TrainDataset(Dataset):
    random.seed(1997)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, cfg, phase='train'):
        self.sdf = cfg.sdf
        self.opt = cfg.dataset
        self.cfg = cfg
        self.syn_dir = self.opt.syn_dir
        self.sample_dir = self.opt.sample_dir
        self.smpl_fit_dir = self.opt.smpl_fit_dir
        self.overfit = self.opt.overfit
        self.is_train = (phase == 'train')
        self.sigma = self.opt.sigma
        self.load_size = self.opt.input_size
        self.num_views = self.opt.num_views
        self.b_min = np.array(self.opt.b_min)
        self.b_max = np.array(self.opt.b_max)
        self.n_rotation = self.opt.train_n_rotation if self.is_train else self.opt.test_n_rotation
        self.n_action = self.opt.train_n_action if self.is_train else self.opt.test_n_action
        # self.rotations = np.arange(0, 360, 360 // self.n_rotation)
        self.actions = np.asarray(range(self.n_action))

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=self.opt.aug_bri,
                                   contrast=self.opt.aug_con,
                                   saturation=self.opt.aug_sat,
                                   hue=self.opt.aug_hue)
        ]) if (self.opt.aug and not self.opt.input_im == 'normal') else None

        self.subjects = ['197019'] if self.overfit else self.get_subjects()

        self.smpl_model = smpl.create(cfg.smpl.path,
                                      model_type=cfg.smpl.model_type,
                                      gender=cfg.smpl.gender,
                                      use_face_contour=cfg.smpl.use_face_contour)
        self.smpl_faces = np.array(self.smpl_model.faces).astype(np.int16)

    def get_subjects(self):
        if self.is_train:
            subjects = sorted(Path('./splits/train100.txt').read_text().strip().split('\n'))
        else:
            subjects = sorted(Path('./splits/test50.txt').read_text().strip().split('\n'))[:20]
        return subjects

    def __len__(self):
        return len(self.subjects) * self.n_action * self.n_rotation // self.num_views

    def get_render(self, sid):
        calib_list = []
        render_list = []
        joint_transform_list = []
        extrinsic_list = []
        intrinsic_list = []
        poses_list = []
        aids = np.random.choice(os.listdir(os.path.join(self.syn_dir, sid)), self.num_views, replace=False)

        for aid in aids:
            if self.opt.input_im == 'rgb':
                render_path = random.choice(glob('%s/%s/%s/RENDER/*.png' % (self.syn_dir, sid, aid)))
            else:
                render_path = random.choice(glob('%s/%s/%s/NORMAL/*.png' % (self.syn_dir, sid, aid)))

            rid = render_path.split('/')[-1][:-4]
            mask_path = '%s/%s/%s/MASK/%s.png' % (self.syn_dir, sid, aid, rid)
            param_path = '%s/%s/%s/PARAM/%s.npz' % (self.syn_dir, sid, aid, rid)

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param['ortho_ratio']
            # world unit / model unit
            # replace the scale and center of normalizing canonical mesh with posed mesh
            scale = param['scale']
            # camera center world coordinate
            center = param['center']
            # model rotation
            R = param['R']
            # joint transform
            jointT = torch.from_numpy(param['jointT'])
            pose = torch.from_numpy(param['pose'])
            # with torch.no_grad():
            #     output = self.smpl_model(global_orient=pose[None, :1], body_pose=pose[None, 1:], custom_out=True)
            #     jointT = output.joint_transform[:, :24]

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
            uv_intrinsic[0, 0] = 1.0 / float(self.load_size // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.load_size // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.load_size // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            # Transform under image pixel space
            render = Image.open(render_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size)
                mask = ImageOps.expand(mask, pad_size)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.85, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.BILINEAR)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1
                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)), int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)), int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] -= dx / float(self.load_size // 2)
                trans_intrinsic[1, 3] -= dy / float(self.load_size // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                if self.aug_trans is not None:
                    render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            # Note that here we replace the origin extrinsic calib in posed space by canonical space
            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.from_numpy(np.matmul(intrinsic, extrinsic)).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            poses_list.append(pose)
            joint_transform_list.append(jointT)
            intrinsic_list.append(torch.from_numpy(intrinsic))
            extrinsic_list.append(torch.from_numpy(extrinsic))

        return {
            # 'aids': aids,
            'image': torch.stack(render_list).float(),
            'calib': torch.stack(calib_list).float(),
            'poses': torch.stack(poses_list).float(),
            'joint_transform': torch.stack(joint_transform_list).float()
        }

    @torch.no_grad()
    def load_smpl(self, subject, poses):
        """
        Args:
            subject: str
            poses: (torch.Tensor) [num_view, 24, 3]
        Returns:

        """

        self.smpl_model = self.smpl_model.cpu()

        # canonical smpl data
        smpl_data = np.load(os.path.join(self.smpl_fit_dir, '../data_sample', subject, 'smpl/param.npz'))
        betas = torch.from_numpy(smpl_data['betas']).view(1, -1).contiguous()

        betas = betas.expand(poses.shape[0], -1).contiguous()

        if self.cfg.smpl.model_type == 'smpl_vitruvian':
            canon_vertices, canon_joints = self.smpl_model.initiate_vitruvian(
                vitruvian_angle=self.cfg.smpl.vitruvian_angle,
                custom_out=True,
                betas=betas)
        else:
            smpl_out = self.smpl_model(betas=betas, custom_out=True)
            canon_vertices, canon_joints = smpl_out.vertices, smpl_out.joints

        # posed smpl data
        posed_smpl_verts = []
        posed_joints = []
        for pose in poses:
            posed_out = self.smpl_model(
                betas=betas,
                global_orient=pose[None, :1],
                body_pose=pose[None, 1:],
                custom_out=True)
            posed_smpl_verts.append(posed_out.vertices)
            posed_joints.append(posed_out.joints[:, :24])

        return {
            'canon_smpl_vert': canon_vertices[0].float(),  # [N, 3]
            'posed_smpl_vert': torch.cat(posed_smpl_verts).float(),  # [num_view, N, 3]
            'smpl_lbs_weights': self.smpl_model.lbs_weights,  # [N, 24]
            'canon_smpl_joints': canon_joints[0, :24].t(),  # [N, 24]
            'posed_smpl_joints': torch.cat(posed_joints).transpose(1, 2),  # [B, N, 24]
            'smpl_faces': torch.as_tensor(self.smpl_faces).long()  # [N, 3]
        }

    def sampling_occ_cache(self, subject, joint_transform, calib):
        sample_data = torch.load(os.path.join(self.sample_dir, f'{subject}.pt'))
        samples = torch.cat([sample_data['perturb_points'], sample_data['bbox_points']])
        labels = torch.cat([sample_data['perturb_sdf'], sample_data['bbox_sdf']])
        weights = torch.cat([sample_data['perturb_weight'], sample_data['bbox_weight']])

        # inside
        n_half = (self.opt.num_surface + self.opt.num_perturb + self.opt.num_bbox) // 2
        inside = labels.squeeze().numpy() < 0
        ids = np.random.randint(0, inside.sum(), n_half)
        pos_points = samples[inside][ids]
        pos_weight = weights[inside][ids]

        outside = np.logical_not(inside)
        ids = np.random.randint(0, outside.sum(), n_half)
        neg_points = samples[outside][ids]
        neg_weight = weights[outside][ids]

        canon_points = torch.cat([pos_points, neg_points])
        weights = torch.cat([pos_weight, neg_weight])
        labels = torch.cat([torch.ones(1, n_half), torch.zeros(1, n_half)], 1)

        posed_points, projected_points = warp_and_project_points(canon_points, weights, joint_transform, calib)

        return {
            'canon_points': canon_points.t().float(),  # [3, N]
            'posed_points': posed_points.transpose(1, 2).float(),  # [num_view, 3, num_surf]
            'projected_points': projected_points.transpose(1, 2).float(),  # [num_view, 3, num_surf]
            'labels': labels,  # [1, N]
        }

    def sampling_sdf_cache(self, subject, joint_transform, calib):
        sample_data = torch.load(os.path.join(self.sample_dir, f'{subject}.pt'))
        ids = np.random.randint(0, len(sample_data['surface_points']), self.opt.num_surface)
        surf_points = sample_data['surface_points'][ids]
        surf_weight = sample_data['surface_weight'][ids]
        surf_normal = sample_data['surface_normal'][ids]

        ids = np.random.randint(0, len(sample_data['perturb_points']), self.opt.num_perturb)
        perturb_points = sample_data['perturb_points'][ids]
        perturb_weight = sample_data['perturb_weight'][ids]
        perturb_sdf = sample_data['perturb_sdf'][ids]

        ids = np.random.randint(0, len(sample_data['bbox_points']), self.opt.num_bbox)
        bbox_points = sample_data['bbox_points'][ids]
        bbox_weight = sample_data['bbox_weight'][ids]
        bbox_sdf = sample_data['bbox_sdf'][ids]

        canon_points = torch.cat([surf_points, perturb_points, bbox_points])
        weights = torch.cat([surf_weight, perturb_weight, bbox_weight])
        labels = torch.cat([torch.zeros(self.opt.num_surface, 1), perturb_sdf, bbox_sdf])

        posed_points, projected_points = warp_and_project_points(canon_points, weights, joint_transform, calib)

        return {
            'canon_points': canon_points.t().float(),  # [3, N]
            'posed_points': posed_points.transpose(1, 2).float(),  # [num_view, 3, num_surf]
            'projected_points': projected_points.transpose(1, 2).float(),  # [num_view, 3, num_surf]
            'canon_surf_normal': surf_normal.t().float(),  # [3, num_surf]
            'labels': labels.t(),  # [1, N]
        }

    def sampling_sdf_from_mesh(self, subject, joint_transform, calib):
        obj_file = os.path.join(self.opt.obj_dir, subject, 'da-pose.obj')
        skin_file = os.path.join(self.opt.obj_dir, subject, 'skin_weight.npz')

        canon_mesh = trimesh.load(obj_file, **{'process': False, 'maintain_order': True})
        skin_weights = np.load(skin_file)['skin_weight']

        # sampling on surface
        num_surface = self.opt.num_surface
        N = canon_mesh.vertices.shape[0]
        ids = np.random.choice(np.arange(N), num_surface, p=np.ones(N).astype(np.float32) / N)
        # ids = np.random.randint(0, canon_mesh.vertices.shape[0], num_surface)
        canon_surf_points = canon_mesh.vertices[ids]
        canon_surf_normal = canon_mesh.vertex_normals[ids]

        # sampling off surface
        perturb_points = canon_surf_points + np.random.normal(scale=self.opt.sigma, size=(num_surface, 3))
        bbox_points = np.random.rand(num_surface // 8, 3) * (self.b_max - self.b_min) + self.b_min
        off_surf_points = np.concatenate([perturb_points, bbox_points])
        off_weights = query_lbs_weight(off_surf_points, canon_mesh.vertices, skin_weights, device=torch.device('cuda:0')
                                       ).cpu().numpy()

        # warp and project
        cannon_points = np.concatenate([canon_surf_points, off_surf_points])
        weights = np.concatenate([skin_weights[ids], off_weights])
        posed_points, projected_points = warp_and_project_points(cannon_points, weights, joint_transform, calib)

        # surface normal in posed space
        posed_points = warp_and_project_points(canon_mesh.vertices, skin_weights, joint_transform)
        posed_surf_normal = np.stack(
            [trimesh.Trimesh(pts, canon_mesh.faces, **{'process': False, 'maintain_order': True}).vertex_normals[ids]
             for pts in posed_points.numpy()])

        return {
            'canon_points': torch.as_tensor(cannon_points).t().float(),  # [3, N]
            'posed_points': posed_points.transpose(1, 2).float(),  # [num_view, 3, num_surf]
            'projected_points': projected_points.transpose(1, 2).float(),  # [num_view, 3, num_surf]
            'canon_surf_normal': torch.as_tensor(canon_surf_normal).t().float(),  # [3, num_surf]
            'posed_surf_normal': torch.as_tensor(posed_surf_normal).transpose(1, 2).float(),  # [3, num_surf]
        }

    def get_item(self, index):
        sid = index % len(self.subjects)
        sid = self.subjects[sid]
        data_dict = {
            'sid': sid,
            'b_min': self.b_min,
            'b_max': self.b_max,
        }
        data_dict.update(self.get_render(sid))
        data_dict.update(self.load_smpl(sid, data_dict['poses']))

        if self.sdf:
            data_dict.update(self.sampling_sdf_cache(sid, data_dict['joint_transform'], data_dict['calib']))
        else:
            data_dict.update(self.sampling_occ_cache(sid, data_dict['joint_transform'], data_dict['calib']))
        return data_dict

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            print(colored(e, "red"))
            return self.get_item(random.randint(0, self.__len__() - 1))
