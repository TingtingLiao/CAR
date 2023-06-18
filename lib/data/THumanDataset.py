import random
import logging
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from .mesh_util import *
from lib.data.mesh_util import projection
from lib.renderer.mesh import load_fit_body, get_smpl_model
from lib.common.geometry import index

log = logging.getLogger('trimesh')
log.setLevel(40)


class THuman(Dataset):
    random.seed()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, cfg, phase='train'):
        self.opt = cfg.dataset
        self.syn_dir = self.opt.thuman_syn_dir
        self.data_dir = self.opt.thuman_data_dir
        self.overfit = self.opt.overfit
        self.sdf = cfg.sdf
        self.is_train = (phase == 'train')
        self.sigma = self.opt.sigma
        self.input_size = self.opt.input_size
        self.num_views = self.opt.num_views
        self.scale = 100
        self.b_min = np.array(self.opt.b_min)
        self.b_max = np.array(self.opt.b_max)
        self.rotations = np.arange(0, 360) if self.is_train else [0]
        self.use_depth = cfg.net.use_depth

        # self.noise_dict = self.opt.noise_dict

        # PIL to tensor
        self.image_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # PIL to tensor
        self.mask_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=self.opt.aug_bri,
                                   contrast=self.opt.aug_con,
                                   saturation=self.opt.aug_sat,
                                   hue=self.opt.aug_hue)
        ]) if self.opt.aug and not self.opt.input_im == 'normal' else None

        self.subjects = self.get_subjects()
        self.smpl_model = get_smpl_model('smplx', 'male')
        self.faces = self.smpl_model.faces.astype(np.int32)
        self.smpl_noise = False

    def get_subjects(self):
        subjects = sorted(Path(os.path.join(self.data_dir, 'all.txt')).read_text().strip().split('\n'))
        if self.overfit:
            sid = subjects[0]
            self.mesh = self.load_mesh(f'{self.data_dir}/scans/{sid}/{sid}.obj')
            return [sid]
        elif self.is_train:
            return subjects[:400]
        else:
            return subjects[400:]

    def __len__(self):
        return len(self.subjects) * len(self.rotations)

    def vis_debug(self, mesh, image_path=None):
        from lib.common.render import Render
        mesh.vertices[:, 1] *= -1
        render = Render(device=torch.device('cuda'))
        render.load_mesh(mesh.vertices, mesh.faces)
        render_geo = render.get_image(cam_ids=[0]) * 255
        if image_path is not None:
            rgb = cv2.imread(image_path)
            render_geo = (render_geo * 0.5 + rgb * 0.5)
        cv2.imwrite('/media/liaotingting/usb3/im.png', render_geo)
        exit()

    def load_mesh(self, mesh_path):
        mesh = trimesh.load(mesh_path, **{'skip_materials': True, 'process': False, 'maintain_order': True})
        mesh.vertices *= self.scale
        return mesh

    def sampling_occ_mesh(self, mesh_path, calib):
        mesh = self.load_mesh(mesh_path)
        surf_samples, surf_ids = trimesh.sample.sample_surface(mesh, self.opt.num_surface * 4)
        surf_normals = mesh.face_normals[surf_ids]

        # Sampling offsets are random noise with constant scale (15cm - 20cm)
        offset = np.random.normal(scale=self.opt.sigma * self.scale, size=(self.opt.num_surface * 4, 1))
        samples_surface = surf_samples + surf_normals * offset

        # Uniform samples in [-1, 1]
        calib_inv = np.linalg.inv(calib)
        n_samples_space = self.opt.num_surface // 4
        samples_space = 2.0 * np.random.rand(n_samples_space, 3) - 1.0
        samples_space = projection(samples_space, calib_inv)

        samples = np.concatenate([samples_surface, samples_space], 0)

        np.random.shuffle(samples)

        inside = mesh.contains(samples)
        inside_samples = samples[inside >= 0.5]
        outside_samples = samples[inside < 0.5]

        nin = inside_samples.shape[0]
        if nin > self.opt.num_surface // 2:
            inside_samples = inside_samples[:self.opt.num_surface // 2]
            outside_samples = outside_samples[:self.opt.num_surface // 2]

        else:
            outside_samples = outside_samples[:(self.opt.num_surface - nin)]

        samples = torch.from_numpy(np.concatenate([inside_samples, outside_samples])).float()
        samples = projection(samples, calib)

        labels = np.concatenate([np.zeros(inside_samples.shape[0]), np.ones(outside_samples.shape[0])])

        return {
            'projected_points': samples.t().unsqueeze(0),
            'labels': torch.from_numpy(labels).unsqueeze(0).float()
        }

    def sampling_sdf_mesh(self, mesh_path, cache_file, calib):
        mesh = self.load_mesh(mesh_path) if not self.overfit else self.mesh
        mesh.vertices = projection(mesh.vertices, calib.numpy())
        mesh.faces = mesh.faces[:, [1, 0, 2]]

        # surface samples
        surf_samples, surf_ids = trimesh.sample.sample_surface(mesh, self.opt.num_surface)
        surf_normals = mesh.face_normals[surf_ids]

        # around surface samples
        data = torch.load(cache_file)
        N = self.opt.num_perturb // 2
        pos_pts = torch.stack(random.choices(data['pos_pts'], k=N))
        neg_pts = torch.stack(random.choices(data['neg_pts'], k=N))
        samples_perturb = torch.cat([pos_pts, neg_pts]).float()
        samples_perturb = projection(samples_perturb, calib)

        # Uniform samples in [-1, 1]
        samples_bbox = 2.0 * np.random.rand(self.opt.num_bbox, 3) - 1.0
        inside = mesh.contains(samples_bbox)

        samples = np.concatenate([surf_samples, samples_perturb, samples_bbox])
        labels = torch.cat([torch.ones(self.opt.num_surface) * 0.5,
                            torch.zeros(N), torch.ones(N),
                            1 - torch.as_tensor(inside).float()])
        # inside_ids = labels.numpy() > 0.5
        # print(labels.shape, samples.shape)
        # print(np.sum(inside_ids))
        # save_obj_mesh('/media/liaotingting/usb3/smpl.obj', samples[inside_ids])
        # exit()

        return {
            'projected_points': torch.as_tensor(samples).t().float().unsqueeze(0),
            'projected_surf_normal': torch.as_tensor(surf_normals).t().float().unsqueeze(0),
            'labels': labels.unsqueeze(0)
        }

    def sampling_occ_cache(self, mesh_path, cache_file, calib):
        mesh = self.load_mesh(mesh_path) if not self.overfit else self.mesh
        mesh.vertices = projection(mesh.vertices, calib.numpy())
        mesh.faces = mesh.faces[:, [1, 0, 2]]

        data = torch.load(cache_file)
        N = self.opt.num_surface
        pos_pts = torch.stack(random.choices(data['pos_pts'], k=N))
        neg_pts = torch.stack(random.choices(data['neg_pts'], k=N))
        samples = torch.cat([pos_pts, neg_pts]).float()
        samples = projection(samples, calib)

        # Uniform samples in [-1, 1]
        samples_bbox = 2.0 * np.random.rand(self.opt.num_bbox, 3) - 1.0
        bbox_inside = mesh.contains(samples_bbox)
        samples = np.concatenate([samples, samples_bbox])
        labels = 1 - torch.cat([torch.ones(N), torch.zeros(N),
                                torch.as_tensor(bbox_inside).float()])

        return {
            'projected_points': torch.as_tensor(samples).t().float().unsqueeze(0),
            'labels': labels.unsqueeze(0)
        }

    def compute_smpl_verts(self, data_dict):
        noise_dict = None
        if self.smpl_noise:
            smplx_param = np.load(data_dict['smplx_path'], allow_pickle=True)
            smplx_pose = smplx_param["body_pose"]  # [1,63]
            smplx_betas = smplx_param["betas"]  # [1,10]
            smplx_pose, smplx_betas = add_noise(
                'smplx',
                smplx_pose[0],
                smplx_betas[0],
                self.noise_dict,
                hashcode=(hash(f"{data_dict['sid']}_{data_dict['rid']}")) % (10 ** 8))
            noise_dict = dict(betas=smplx_betas, body_pose=smplx_pose)

        smplx_out, _ = load_fit_body(fitted_path=data_dict['smplx_path'],
                                     scale=self.scale,
                                     smpl_model=self.smpl_model,
                                     noise_dict=noise_dict)
        return smplx_out

    def load_smpl(self, data_dict):
        smpl_body = self.compute_smpl_verts(data_dict)
        smplx_verts = projection(smpl_body.vertices, data_dict['calib']).float().numpy()

        return {
            'projected_smpl_vert': torch.as_tensor(smplx_verts).unsqueeze(0),
            'smpl_faces': torch.as_tensor(self.faces[:, [1, 0, 2]]).long()
        }

    def load_calib(self, calib_path):
        calib_data = np.loadtxt(calib_path)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        return {'calib': torch.from_numpy(calib_mat).float()}

    def load_image(self, path, channel=3):
        rgba = Image.open(path).convert('RGBA')
        mask = rgba.split()[-1]
        image = rgba.convert('RGB')
        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)
        image = (image * mask)[:channel]
        return image.unsqueeze(0)

    def load_depth(self, path):
        depth = Image.open(path).convert('RGBA')
        mask = depth.split()[-1]
        depth = depth.convert('L')
        depth = self.image_to_tensor(depth)
        mask = self.mask_to_tensor(mask)
        depth = depth * mask + (mask - 1)
        return depth.unsqueeze(0)

    def get_item(self, index):
        rid = index % len(self.rotations)
        sid = index // len(self.rotations)
        sid = self.subjects[sid]
        rid = self.rotations[rid]

        mesh_path = f'{self.data_dir}/scans/{sid}/{sid}.obj'
        smplx_path = f'{self.data_dir}/fits-smplx/{sid}/smplx_param.pkl'
        calib_path = f'{self.syn_dir}/{sid}/CALIB/{rid:03d}.txt'
        sample_path = f'{self.data_dir}/samples/{sid}.pt'
        fv_image_path = f'{self.syn_dir}/{sid}/NORMAL_F/{rid:03d}.png'
        bv_image_path = f'{self.syn_dir}/{sid}/NORMAL_B/{rid:03d}.png'

        # fv_depth_path = f'{self.syn_dir}/{sid}/T_DEPTH_F/{rid:03d}.png'
        # bv_depth_path = f'{self.syn_dir}/{sid}/T_DEPTH_B/{rid:03d}.png'

        # gt depth
        #  if self.is_train else 'Pred_DEPTH_F'
        dir_name = 'DEPTH_F'
        fv_depth_path = f'{self.syn_dir}/{sid}/{dir_name}/{rid:03d}.png'
        bv_depth_path = f'{self.syn_dir}/{sid}/{dir_name}/{rid:03d}.png'

        # try:
        data_dict = {
            'sid': sid,
            'rid': rid,
            'b_min': self.b_min,
            'b_max': self.b_max,
            'scale': self.scale,
            'smplx_path': smplx_path,
            'fv_depth_path': fv_depth_path,
            'bv_depth_path': bv_depth_path,
        }

        # load image
        data_dict.update({'image': self.load_image(fv_image_path)})

        # load calib
        data_dict.update(self.load_calib(calib_path))

        # load smpl
        data_dict.update(self.load_smpl(data_dict))

        if self.sdf:
            data_dict.update(self.sampling_sdf_mesh(mesh_path, sample_path, data_dict['calib']))
        else:
            data_dict.update(self.sampling_occ_cache(mesh_path, sample_path, data_dict['calib']))

        if self.use_depth:
            data_dict.update({
                'depth': torch.cat([-self.load_depth(fv_depth_path), self.load_depth(bv_depth_path)], 1)
            })

        return data_dict
        # except Exception as e:
        #     print(e, sid)
        #     exit()
        #     return self.get_item(idx=random.randint(0, self.__len__() - 1))

    def get_sdf(self, data_dict):
        fv_depth = -self.load_depth(data_dict['fv_depth_path'])
        bv_depth = self.load_depth(data_dict['bv_depth_path'])

        pts = data_dict['projected_points']
        xy, z = torch.split(pts, [2, 1], dim=1)

        fv_depth = index(fv_depth, xy)
        bv_depth = index(bv_depth, xy)

        debug = False
        if debug:
            f_pts = torch.cat([xy, fv_depth], 1)
            b_pts = torch.cat([xy, bv_depth], 1)
            pts = torch.cat([f_pts, b_pts], 2)[0].numpy().T
            save_obj_mesh('/media/liaotingting/usb3/smpl.obj', pts)

        sdf = torch.cat([z - fv_depth, bv_depth - z], 1)

        return {'sdf': sdf}

    def __getitem__(self, index):
        return self.get_item(index)
