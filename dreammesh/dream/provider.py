import os
import cv2
import glob
import json
from tqdm import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import utils as util
from .obj import Mesh
from .renderer import Renderer

DIR_COLORS = np.array(
    [
        [255, 0, 0, 255],  # front
        [0, 255, 0, 255],  # side
        [0, 0, 255, 255],  # back
        [255, 255, 0, 255],  # side
        [255, 0, 255, 255],  # overhead
        [0, 255, 255, 255],  # bottom
    ],
    dtype=np.uint8)


def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [-half_front, half_front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 1  [180+front, 360)
    # top = 3           [0, overhead]
    # bottom = 4        [180-overhead, 180]

    half_front = front / 2.
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    # res[(phis < half_front) & (phis > 2 * np.pi - half_front)] = 0
    res[(phis >= half_front) & (phis < np.pi - half_front)] = 1
    res[(phis >= np.pi - half_front) & (phis < np.pi + half_front)] = 2
    res[(phis >= np.pi + half_front) & (phis < 2 * np.pi - half_front)] = 1
    # override by thetas
    # res[thetas <= overhead] = 3
    # res[thetas >= (np.pi - overhead)] = 4
    return res


def rand_poses(size,
               device,
               radius_range=[1.2, 1.4],
               theta_range=[80, 100],
               phi_range=[0, 360],
               return_dirs=False,
               angle_overhead=30,
               angle_front=60,
               jitter=False,
               uniform_sphere_rate=0.5):
    """
    generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_range: [min, max] camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
        return_dirs: bool return camera direction if true
        angle_overhead: float
        angle_front: float
        jitter: bool
        uniform_sphere_rate: float should be in [0, 1]
    Return:
        poses: [size, 4, 4]
    """
    # if head, we only render front view
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ],
                dim=-1),
            p=2,
            dim=1)
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ],
            dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = util.safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = util.safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = util.safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        angle_overhead = np.deg2rad(angle_overhead)
        angle_front = np.deg2rad(angle_front)
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


def near_head_poses(size,
                    device,
                    radius_range=[0.15, 0.2],
                    theta_range=[85, 95],
                    phi_range=[-60, 60],
                    return_dirs=False,
                    angle_overhead=30,
                    angle_front=60,
                    jitter=False,
                    head_shift=[0, 0.45, 0]):
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    shift = torch.as_tensor(head_shift, device=device).view(1, 3)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) + shift   # [B, 3]
    targets = torch.zeros_like(centers) + shift

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = util.safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = util.safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = util.safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        angle_overhead = np.deg2rad(angle_overhead)
        angle_front = np.deg2rad(angle_front)
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
        dirs += 3
    else:
        dirs = None

    return poses, dirs


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ],
        dim=-1)  # [B, 3]

    # lookat
    forward_vector = util.safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = util.safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = util.safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


class ViewDataset(torch.utils.data.Dataset):

    def __init__(self, opt, device, type='train', size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        if type == 'train':
            self.H = opt.h
            self.W = opt.w
        else:
            self.H = opt.H
            self.W = opt.H
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.opt.min_near
        self.far = 1000  # infinite

        self.aspect = self.W / self.H
        self.full_body = True

        # [debug] visualize poses
        # self.test_camera()

    @staticmethod
    def modify_commandline_options(parser, full_body):
        return parser

    def test_camera(self):
        from dream.obj import Mesh
        from dream.renderer import Renderer
        from dream.utils import plot_grid_images
        mesh = Mesh.load_obj("data/mesh.obj", init_empty_tex=True)

        render = Renderer()

        data = self.collate(list(range(100)))
        normals, alpha = render(mesh, data['mvp'], 512, 512, None, 1, "normal")
        normals = normals.cpu().numpy()
        views = ['front', 'side', 'back', 'side', 'overhead', 'bottom']
        for i, dir in enumerate(data['dir']):
            cv2.putText(normals[i], views[dir], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, color=(255, 255, 255), thickness=6)

        plot_grid_images(normals, 10, 10, "person.png")
        exit()

    def __getitem__(self, idx):
        if self.training:
            # random pose on the fly
            if self.full_body:
                poses, dirs = rand_poses(
                    1,
                    self.device,
                    radius_range=self.opt.radius_range,
                    return_dirs=self.opt.dir_text,
                    angle_overhead=self.opt.angle_overhead,
                    angle_front=self.opt.angle_front,
                    jitter=self.opt.jitter_pose,
                    uniform_sphere_rate=self.opt.uniform_sphere_rate)
            else:
                poses, dirs = near_head_poses(
                    1,
                    self.device,
                    return_dirs=self.opt.dir_text,
                    angle_overhead=self.opt.angle_overhead,
                    angle_front=self.opt.angle_front,
                    jitter=self.opt.jitter_pose)

            # random focal
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
        else:
            # circle pose
            phi = (idx / self.size) * 360
            poses, dirs = circle_poses(
                self.device,
                radius=(self.opt.radius_range[0] + self.opt.radius_range[1]) * 0.5,
                theta=60,
                phi=phi,
                return_dirs=self.opt.dir_text,
                angle_overhead=self.opt.angle_overhead,
                angle_front=self.opt.angle_front)

            # fixed focal
            fov = (self.opt.fovy_range[1] + self.opt.fovy_range[0]) / 2

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2 * focal / self.W, 0, 0, 0],
            [0, -2 * focal / self.H, 0, 0],
            [0, 0, -(self.far + self.near) / (self.far - self.near),
             -(2 * self.far * self.near) / (self.far - self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device)  # yapf: disable

        mvp = projection @ torch.inverse(poses.squeeze(0))

        data = {
            'H': self.H,
            'W': self.W,
            'mvp': mvp,  # [4, 4]
            'poses': poses.squeeze(0),
            'intrinsics': torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device),
            'dir': dirs,
        }

        return data

    def __len__(self):
        return self.size


class MeshDataset:
    def __init__(self, FLAGS, validate=False, batch=10, size=100):
        super().__init__()
        # Init
        self.cam_radius = FLAGS.radius
        self.FLAGS = FLAGS
        self.validate = validate
        self.fovy = np.deg2rad(45)
        if self.validate:
            self.h, self.w = self.FLAGS.H, self.FLAGS.W
        else:
            self.h, self.w = self.FLAGS.h, self.FLAGS.w
        self.aspect = self.w / self.h

        # Load environment map texture
        # self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
        assert os.path.exists(f"{FLAGS.workspace}/mesh/mesh_albedo.png")

        self.ref_mesh = Mesh.load_obj(path=f"{FLAGS.workspace}/mesh/mesh.obj",
                                      albedo_path=f"{FLAGS.workspace}/mesh/mesh_albedo.png"
                                      )

        self.renderer = Renderer()
        print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (
            self.ref_mesh.f.shape[0], self.ref_mesh.v.shape[0]))

        self.size = size
        self.batch = batch

        self.prepare_data(100)
        exit()
        # from NormalNet import get_normal_model
        # self.NormalNet = get_normal_model().cuda()

    @torch.no_grad()
    def get_normal(self, image, mask):
        nmlF, _ = self.NormalNet(image.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2))
        nmlF = F.normalize(nmlF, eps=1e-6)
        return nmlF.permute(0, 2, 3, 1)[..., [2, 1, 0]]

    def prepare_data(self, num_frame):
        self.validate = True
        out_dir = self.FLAGS.workspace
        print(out_dir + "/image")
        os.makedirs(out_dir + "/image", exist_ok=True)
        import imageio

        prompt_list = {
            "superman": "superman, full body",
            "Deadpool": "Deadpool, full body",
            "batman": "Batman, full body",
            "antman": "Ant man, full body",

            "trump": "Donald Trump, full body",
            "lincoln": "Abraham Lincoln, full body",
            "obama": "Barack Obama, full body",
            "clinton": "Hilary Clinton",

            "gardener": "Gardener, full body",
            "robot": "Robot, full body",
            "witch": "Witch, full body",
            "wizard": "Wizard, full body",
        }
        row = 3
        col = 4
        mesh_list = [Mesh.load_obj(
            # path=f"../nvdiffrec/out/{name}/dmtet_mesh/mesh.obj",
            path=f"experiments/{name}/mesh/mesh.obj",
            albedo_path=f"../nvdiffrec/out/{name}/dmtet_mesh/texture_kd.png"
        )
            for name in prompt_list.keys()
        ]

        video_list = []
        for i in tqdm(range(num_frame)):
            mv, mvp, campos = self._rotate_scene(i)

            images = []
            for mesh in mesh_list:
                rgb, alpha = self.renderer(mesh, mvp, self.h, self.w, None, 1, "normal")
                rgb = rgb * alpha + (1 - alpha)
                img = rgb[0, ...].detach().cpu().numpy()
                # util.save_image(f"{out_dir}/image/albedo.png",
                #                 np.concatenate([img, alpha[0, ...].detach().cpu().numpy()], axis=-1))

                images.append((img * 255).astype(np.uint8))
            images = np.vstack([np.hstack(images[r * col:(r + 1) * col]) for r in range(row)])

            video_list.append(images)
        imageio.mimsave(f"./experiments/geo4.mp4", np.stack(video_list), fps=20, quality=8, macro_block_size=1)

    def _rotate_scene(self, itr=None):
        if self.validate:
            ang_x = -0.4
            ang_y = (itr / 50) * np.pi * 2
        else:
            ang_y = random.random() * np.pi * 2
            ang_x = (random.random() - 0.5) * np.pi * 0.1

        proj_mtx = util.perspective(self.fovy, self.aspect, self.FLAGS.min_near, 1000)

        # Smooth rotation for display.
        mv = util.translate(0, 0, -self.cam_radius) @ util.rotate_x(ang_x) @ util.rotate_y(ang_y)
        mvp = proj_mtx @ mv
        poses = torch.inverse(mv)

        return mv[None].cuda(), mvp[None].cuda(), poses[None].cuda()

    def collate(self, index):
        mvp_list = []
        poses_list = []
        img_list = []
        for idx in index:
            mv, mvp, poses = self._rotate_scene(idx)

            rgb, alpha, _ = self.renderer(self.ref_mesh, mvp, self.h, self.w, None, 1, "albedo")

            rgb = rgb + (1 - alpha)
            mvp_list.append(mvp)
            poses_list.append(poses)
            img_list.append(torch.cat([rgb, alpha], -1))
        focal = self.h / (2 * np.tan(self.fovy / 2))
        intrinsics = np.array([focal, focal, self.h / 2, self.w / 2])

        data = {
            'H': self.h,
            'W': self.w,
            'intrinsics': intrinsics,
            'mvp': torch.cat(mvp_list),
            'poses': torch.cat(poses_list),
            'img': torch.cat(img_list)
        }

        # data['normal'] = self.get_normal(data['img'][..., :3], data['img'][..., 3:])

        return data

    def dataloader(self):
        return DataLoader(list(range(self.size)), batch_size=self.batch, collate_fn=self.collate, shuffle=True)


