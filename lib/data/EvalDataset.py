import random
import logging
from pathlib import Path
from .mesh_util import *
from PIL import Image
import imageio
from tqdm import tqdm
import smpl
from skimage import img_as_ubyte
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from lib.pymaf.models import pymaf_net
from lib.pymaf.core import path_config
from lib.pymaf.utils.imutils import get_transformer
from lib.icon.model.NormalNet import get_icon_NormalNet
from lib.data.mesh_util import compute_normal, linear_blend_skinning
from lib.model.geometry import orthogonal
from glob import glob
from lib.common.render import Render

log = logging.getLogger('trimesh')
log.setLevel(40)


class EvalDataset():
    def __init__(self, data_name, cfg, device):
        random.seed(1993)
        self.data_name = data_name
        self.cfg = cfg
        self.opt = cfg.dataset
        self.image_size = self.opt.input_size
        self.syn_dir = self.opt.syn_dir if data_name == 'mvp' else self.opt.buff_dir
        self.obj_dir = self.opt.obj_dir
        self.num_views = self.opt.num_views
        self.b_min = np.array(self.opt.b_min)
        self.b_max = np.array(self.opt.b_max)

        self.image_files = self.get_image_files()
        # self.subjects = sorted(Path('./splits/test50.txt').read_text().strip().split('\n'))[:20]
        # self.rotations = [0]
        # self.actions = np.asarray(range(self.num_views))

        self.device = device
        self.render = Render(device=device)
        self.smpl_model = smpl.create(cfg.smpl.path,
                                      # model_type=cfg.smpl.model_type,
                                      gender=cfg.smpl.gender,
                                      use_face_contour=cfg.smpl.use_face_contour
                                      ).to(self.device)

        self.smpl_faces = self.smpl_model.faces.astype(np.int16)

        self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS).to(self.device)
        self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)['model'], strict=True)
        self.hps.eval()

        self.NormalNet = None

        self.image_to_tensor, self.mask_to_tensor, self.image_to_pymaf_tensor = get_transformer(self.image_size)

        # self.prepare_data()

    def get_image_files(self):
        if self.data_name == 'mvp':
            subjects = sorted(Path('./splits/test50.txt').read_text().strip().split('\n'))[:20]
            return [f'{self.syn_dir}/{sid}/000/RENDER/000.png' for sid in subjects]
        if self.data_name == 'buff':
            return sorted(glob(f'{self.syn_dir}/*/RENDER/*.png'))

    def __len__(self):
        return len(self.image_files)

    def init_normal_net(self):
        self.NormalNet = get_icon_NormalNet()

    def load_gt_obj(self, obj_path, rid=0, param_path=None, skin_weight_path=None, **kwargs):
        mesh = trimesh.load(obj_path, process=False)

        if self.data_name == 'buff':
            if not rid == 0:
                rotate = make_rotate(0, np.radians(rid + 180), 0)
                mesh.vertices = np.matmul(mesh.vertices, rotate.T)
            return mesh

        param = np.load(param_path, allow_pickle=True)
        joint_transform = param['jointT']
        skin_weights = np.load(skin_weight_path)['skin_weight']

        # calib
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
        uv_intrinsic[0, 0] = 1.0 / float(self.image_size // 2)
        uv_intrinsic[1, 1] = 1.0 / float(self.image_size // 2)
        uv_intrinsic[2, 2] = 1.0 / float(self.image_size // 2)
        # Transform under image pixel space
        trans_intrinsic = np.identity(4)
        intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
        calib = torch.from_numpy(np.matmul(intrinsic, extrinsic)).float()[None]

        posed_v = linear_blend_skinning(mesh.vertices[None], skin_weights[None], joint_transform[None])
        projected_v = orthogonal(posed_v.transpose(1, 2), calib).transpose(1, 2)[0].numpy()
        projected_v[:, 1] *= -1
        posed_mesh = trimesh.Trimesh(projected_v, mesh.faces)

        return mesh, posed_mesh

    def load_normal(self, icon_f_nml_path, icon_b_nml_path):
        """
        Args:
            sid: str of subject id
            aid: str of action id
            rid: int of rotation id
            image: FloatTensor [3, H, W]
            mask: FloatTensor [3, H, W]
        Returns:
            nmlF: FloatTensor [3, H, W]
        """
        nmlF = Image.open(icon_f_nml_path).convert('RGB')
        nmlF = self.image_to_tensor(nmlF)

        nmlB = Image.open(icon_b_nml_path).convert('RGB')
        nmlB = self.image_to_tensor(nmlB)

        return nmlF, nmlB

    def load_smpl(self, smpl_file):
        smpl_data = np.load(smpl_file)
        betas = torch.from_numpy(smpl_data['betas']).view(1, -1).float()
        poses = torch.from_numpy(smpl_data['pose']).view(1, 24, 3, 3).float()
        calib = torch.from_numpy(smpl_data['calib']).view(1, 4, 4).float()

        with torch.no_grad():
            if self.cfg.smpl.model_type == 'smpl_vitruvian':
                smpl_model = smpl.create(self.cfg.smpl.path,
                                         model_type=self.cfg.smpl.model_type,
                                         gender=self.cfg.smpl.gender,
                                         use_face_contour=self.cfg.smpl.use_face_contour)

                canon_smpl_v, canon_joints = smpl_model.initiate_vitruvian(
                    vitruvian_angle=self.cfg.smpl.vitruvian_angle,
                    custom_out=True,
                    betas=betas)

            else:
                smpl_model = self.smpl_model.cpu()
                smpl_out = smpl_model(betas=betas, custom_out=True)
                canon_smpl_v, canon_joints = smpl_out.vertices, smpl_out.joints

            posed_output = smpl_model(betas=betas,
                                      global_orient=poses[:, :1],
                                      body_pose=poses[:, 1:],
                                      custom_out=True,
                                      pose2rot=False)
            posed_smpl_v, posed_joints = posed_output.vertices, posed_output.joints

        return {
            'canon_smpl_vert': canon_smpl_v.float(),  # [1, N, 3]
            'posed_smpl_vert': posed_smpl_v.float(),  # [1, N, 6]
            'canon_smpl_joints': canon_joints[:, :24].transpose(1, 2),  # [1, N, 24]
            'posed_smpl_joints': posed_joints[:, :24].transpose(1, 2),  # [1, N, 24]
            'joint_transform': posed_output.joint_transform[:, :24],
            'calib': calib,  # [1, 4, 4]
            'pose': poses
        }

    def optimize_smpl(self, in_tensor, n_iter=100, plot_gif=False):
        # dict_keys(['sid', 'mask', 'betas', 'body_pose', 'global_orient', 'smpl_verts', 'scale', 'trans'])
        """
        Args:
            in_tensor: dict_keys(['sid', 'mask', 'betas', 'body_pose', 'global_orient', 'smpl_verts', 'scale', 'trans'])
            n_iter (int): number of iteration
            plot_gif (bool): save the optimization visual results as gif if true
        Returns:
            calib (torch.Tensor): [1, 4, 4]
            pose (torch.Tensor): [1, 24, 3, 3]
            shape (torch.Tensor): [1, 10]
        """
        optimed_orient = in_tensor['global_orient'].clone().detach()  # [1, 1, 3, 3]
        optimed_pose = in_tensor['body_pose'].clone().detach()  # [1, 23, 3, 3]
        optimed_trans = in_tensor['trans'].clone().detach()  # [1, 3]
        optimed_betas = in_tensor['betas'].clone().detach().mean(0).unsqueeze(0)  # [1, 10]
        scales = in_tensor['scale']

        optimed_pose.requires_grad_()
        optimed_orient.requires_grad_()
        optimed_trans.requires_grad_()
        optimed_betas.requires_grad_()

        batch = len(scales)
        optimizer_smpl = torch.optim.SGD(
            [optimed_pose, optimed_trans, optimed_betas, optimed_orient],
            lr=1e-3,
            momentum=0.9)
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_smpl,
                                                                    factor=0.5,
                                                                    min_lr=1e-5,
                                                                    patience=5)
        out_dir = os.path.dirname(in_tensor['f_nml_save_path'])
        os.makedirs(out_dir, exist_ok=True)
        if plot_gif:
            filename_output = os.path.join(out_dir, 'optimize_smpl.gif')
            writer = imageio.get_writer(filename_output, mode='I', duration=0.05)
        loop_smpl = tqdm(range(n_iter))
        for i in loop_smpl:
            # self.smpl_model = self.smpl_model.to(self.device)
            smpl_out = self.smpl_model(
                betas=optimed_betas,
                body_pose=optimed_pose,
                global_orient=optimed_orient,
                custom_out=True,
                pose2rot=False
            )
            smpl_verts = smpl_out.vertices * scales.unsqueeze(1).expand_as(smpl_out.vertices) + \
                         optimed_trans.unsqueeze(1)
            smpl_verts *= torch.tensor([1.0, -1.0, -1.0]).to(self.device)

            visual_frames = []
            smpl_loss = 0
            for idx in range(batch):
                # render silhouette
                self.render.load_mesh(smpl_verts[idx], self.smpl_faces, use_normal=True)
                T_mask_F, T_mask_B = self.render.get_silhouette_image()

                # silhouette loss
                gt_arr = in_tensor['mask'][idx].to(self.device)

                diff_S = torch.abs(T_mask_F[0] - gt_arr)

                smpl_loss += diff_S.mean()

                visual_frames.append(diff_S.detach().cpu().numpy())

            loop_smpl.set_description(f"Body Fitting = {smpl_loss:.3f}")

            # save to gif file
            visual_frames = np.concatenate(visual_frames, 1)
            if plot_gif:
                writer.append_data(img_as_ubyte(visual_frames))

            # if i in [0, n_iter - 1]:
            #     cv2.imwrite(os.path.join(out_dir, 'iter' + str(i) + '.png'), visual_frames * 255)

            optimizer_smpl.zero_grad()
            smpl_loss.backward(retain_graph=True)
            optimizer_smpl.step()
            scheduler_smpl.step(smpl_loss)

        T_normal_F, T_normal_B = self.render.get_clean_image()
        cv2.imwrite(in_tensor['f_nml_save_path'],
                    (T_normal_B[0].detach().permute(1, 2, 0).cpu().numpy()[..., ::-1] * 0.5 + 0.5) * 255)
        cv2.imwrite(in_tensor['b_nml_save_path'],
                    (T_normal_F[0].detach().permute(1, 2, 0).cpu().numpy()[..., ::-1] * 0.5 + 0.5) * 255)

        calib = torch.stack(
            [torch.FloatTensor(
                [
                    [scale, 0, 0, trans[0]],
                    [0, scale, 0, trans[1]],
                    [0, 0, -scale, -trans[2]],
                    [0, 0, 0, 1],
                ]
            ) for trans, scale in zip(optimed_trans, scales)]
        ).cpu()
        betas = optimed_betas.detach().cpu()
        pose = torch.cat([optimed_orient, optimed_pose], 1).detach().cpu()
        return betas, pose, calib

    def generate_smpl(self, img_path, mask_path, save_dir):
        im_name = img_path.split('/')[-1][:-4]
        f_nml_path = '%s/normal_B_%s.png' % (save_dir, im_name)
        b_nml_path = '%s/normal_F_%s.png' % (save_dir, im_name)
        smpl_file = '%s/param%s.npz' % (save_dir, im_name)

        image_ori = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask = self.mask_to_tensor(mask)
        image_hps = self.image_to_pymaf_tensor(image_ori).unsqueeze(0)

        with torch.no_grad():
            preds_dict = self.hps(image_hps.to(self.device))
            output = preds_dict['smpl_out'][-1]
            scale, tranX, tranY = output['theta'][0, :3]
            trans = torch.tensor([tranX, tranY, 0.0]).to(self.device)

            smpl_data = {
                'mask': mask.squeeze().unsqueeze(0),
                'betas': output['pred_shape'],
                'body_pose': output['rotmat'][:, 1:],
                'global_orient': output['rotmat'][:, 0:1],
                'smpl_verts': output['verts'],
                'scale': scale.view(1, -1),
                'trans': trans.view(1, -1),
                'f_nml_save_path': f_nml_path,
                'b_nml_save_path': b_nml_path
            }
        betas, poses, calib = self.optimize_smpl(smpl_data)
        np.savez(smpl_file, calib=calib.numpy(), betas=betas.numpy(), pose=poses.numpy())

    def generate_icon_normal(self, img_path, mask_path, smpl_dir):
        im_name = img_path.split('/')[-1][:-4]
        f_nml_path = '%s/normal_F_%s.png' % (smpl_dir, im_name)
        b_nml_path = '%s/normal_B_%s.png' % (smpl_dir, im_name)
        icon_f_nml_path = '%s/normal_F_%s_icon.png' % (smpl_dir, im_name)
        icon_b_nml_path = '%s/normal_B_%s_icon.png' % (smpl_dir, im_name)

        if not os.path.exists(f_nml_path) or not os.path.exists(b_nml_path):
            raise FileExistsError(f'{f_nml_path} or {b_nml_path} is not exist.')

        image_ori = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask = self.mask_to_tensor(mask)
        image = self.image_to_tensor(image_ori)
        image = mask.expand_as(image) * image

        f_nml = Image.open(f_nml_path).convert('RGB')
        b_nml = Image.open(b_nml_path).convert('RGB')

        f_nml = self.image_to_tensor(f_nml)
        b_nml = self.image_to_tensor(b_nml)
        in_tensor = {
            'image': image.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'T_normal_F': f_nml.unsqueeze(0),
            'T_normal_B': b_nml.unsqueeze(0),
        }
        if self.NormalNet is None:
            self.init_normal_net()
        with torch.no_grad():
            nmlF, nmlB = self.NormalNet(in_tensor)

        cv2.imwrite(icon_f_nml_path, (nmlF[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 0.5 + 0.5) * 255)
        cv2.imwrite(icon_b_nml_path, (nmlB[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 0.5 + 0.5) * 255)

    def prepare_data(self, img_path, mask_path, smpl_dir):
        im_name = img_path.split('/')[-1][:-4]
        file_names = [f'param{im_name}.npz', f'normal_B_{im_name}.png', f'normal_F_{im_name}.png']
        for file_name in file_names:
            if not os.path.exists(os.path.join(smpl_dir, file_name)):
                self.generate_smpl(img_path, mask_path, smpl_dir)

        file_names = [f'normal_F_{im_name}_icon.png', f'normal_B_{im_name}_icon.png']
        for file_name in file_names:
            if not os.path.exists(os.path.join(smpl_dir, file_name)):
                self.generate_icon_normal(img_path, mask_path, smpl_dir)

    def get_item(self, index):
        img_path = self.image_files[index]
        items = img_path.split('/')
        im_name = items[-1][:-4]
        data_dir = os.path.dirname(os.path.dirname(img_path))

        mask_path = '%s/MASK/%s.png' % (data_dir, im_name)
        param_path = '%s/PARAM/%s.npz' % (data_dir, im_name)
        smpl_path = '%s/SMPL/param%s.npz' % (data_dir, im_name)
        icon_f_nml_path = '%s/SMPL/normal_F_%s_icon.png' % (data_dir, im_name)
        icon_b_nml_path = '%s/SMPL/normal_B_%s_icon.png' % (data_dir, im_name)
        self.prepare_data(img_path, mask_path, smpl_dir=os.path.dirname(smpl_path))

        image_ori = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask = self.mask_to_tensor(mask)
        image = self.image_to_tensor(image_ori)
        image = mask.expand_as(image) * image

        nmlF, nmlB = self.load_normal(icon_f_nml_path, icon_b_nml_path)
        normal = torch.cat([nmlF, nmlB])
        if self.opt.input_im == 'normal':
            image = nmlF

        data_dict = {
            'im_name': im_name,
            'b_min': self.b_min,
            'b_max': self.b_max,
            'image': image.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'normal': normal.unsqueeze(0),
            'smpl_lbs_weights': self.smpl_model.lbs_weights,  # [N, 24]
            'smpl_faces': torch.as_tensor(self.smpl_faces).long(),  # [6890, 3]
            'param_path': param_path,
        }

        if self.data_name == 'mvp':
            sid = data_dir.split('/')[-2]
            data_dict.update({
                'sid': sid,
                'obj_path': os.path.join(self.obj_dir, sid, 'da-pose.obj'),
                'skin_weight_path': os.path.join(self.obj_dir, sid, 'skin_weight.npz'),
            })
        elif self.data_name == 'buff':
            sid = data_dir.split('/')[-1]
            rid = int(im_name.split('_')[-1])
            obj_path = img_path.replace('RENDER', 'OBJ').replace('_'+str(rid), '')[:-4] + '.obj'
            print(obj_path)
            assert os.path.exists(obj_path)
            data_dict.update({'sid': sid, 'rid': rid, 'obj_path': obj_path})

        smpl_dict = self.load_smpl(smpl_path)
        # todo optimize batch with same shape
        smpl_dict['canon_smpl_vert'] = smpl_dict['canon_smpl_vert'][0]
        smpl_dict['canon_smpl_joints'] = smpl_dict['canon_smpl_joints'][0]
        data_dict.update(smpl_dict)

        return data_dict
        # except Exception as e:
        #     print(e)
        #     return self.get_item(random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)


