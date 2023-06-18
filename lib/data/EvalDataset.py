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
from lib.icon.model.NormalNet import get_normal_model
from lib.common.lbs_util import linear_blend_skinning
from lib.common.geometry import orthogonal
from glob import glob
from lib.common.render import Render

log = logging.getLogger('trimesh')
log.setLevel(40)


class EvalDataset():
    def __init__(self, data_name, cfg, device):
        random.seed(2022)
        self.data_name = data_name
        self.cfg = cfg
        self.opt = cfg.dataset
        self.image_size = self.opt.input_size
        if data_name == "mvp":
            self.syn_dir = self.opt.mvp_syn_dir
            self.obj_dir = self.opt.mvp_obj_dir
        elif data_name == "buff":
            self.syn_dir = self.opt.buff_syn_dir
            self.obj_dir = self.opt.buff_obj_dir
        elif data_name == "rp":
            self.syn_dir = self.opt.rp_syn_dir
            self.obj_dir = self.opt.rp_obj_dir

        self.num_views = self.opt.num_views
        self.b_min = np.array(self.opt.b_min)
        self.b_max = np.array(self.opt.b_max)

        self.image_files = self.get_image_files()
        # self.subjects = sorted(Path('./splits/test50.txt').read_text().strip().split('\n'))[:20]
        # self.rotations = [0]
        # self.actions = np.asarray(range(self.num_views))

        self.device = device
        self.render = Render(device=device)
        self.smpl_model = smpl.create(cfg.smpl.model_path,
                                      # model_type=cfg.smpl.model_type,
                                      gender=cfg.smpl.gender,
                                      use_face_contour=cfg.smpl.use_face_contour
                                      ).to(self.device)

        self.smpl_faces = self.smpl_model.faces.astype(np.int16)

        self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS).to(self.device)
        self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)['model'], strict=True)
        self.hps.eval()

        self.normal_type = 'icon'
        self.NormalNet = get_normal_model(self.normal_type).to(self.device)

        self.image_to_tensor, self.mask_to_tensor, self.image_to_pymaf_tensor = get_transformer(self.image_size)

        # self.prepare_data()

    def get_image_files(self):
        if self.data_name == 'mvp':
            subjects = sorted(Path('./splits/test50.txt').read_text().strip().split('\n'))
            return [f'{self.syn_dir}/{sid}/000/RENDER/000.png' for sid in subjects]
        elif self.data_name == 'buff':
            return sorted(glob(f'{self.syn_dir}/*/RENDER/*.png'))
        elif self.data_name == "rp":
            subjects = sorted(Path('./splits/rp.txt').read_text().strip().split('\n'))[50:]
            return [f'{self.syn_dir}/{sid}/000/RENDER/000.png' for sid in subjects]
        else:
            raise ValueError("")

    def __len__(self):
        return len(self.image_files)

    def get_calib(self, param):
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
        return calib

    def load_gt_obj(self, canon_obj_path, posed_obj_path=None, rid=0, param_path=None, skin_weight_path=None, **kwargs):

        if self.data_name == 'buff':
            mesh = trimesh.load(posed_obj_path, process=False)
            if not rid == 0:
                rotate = make_rotate(0, np.radians(rid + 180), 0)
                mesh.vertices = np.matmul(mesh.vertices, rotate.T)
            return mesh

        if self.data_name == "rp":
            c_mesh = trimesh.load(canon_obj_path, process=False)
            p_mesh = trimesh.load(posed_obj_path, process=False)
            param = np.load(param_path, allow_pickle=True)
            calib = self.get_calib(param)

            posed_v = torch.from_numpy(p_mesh.vertices).float()[None]
            projected_v = orthogonal(posed_v.transpose(1, 2), calib).transpose(1, 2)[0].numpy()
            projected_v[:, 1] *= -1
            p_mesh = trimesh.Trimesh(projected_v, p_mesh.faces)

            return c_mesh, p_mesh

        if self.data_name == "mvp":
            mesh = trimesh.load(canon_obj_path, process=False)
            param = np.load(param_path, allow_pickle=True)
            joint_transform = param['jointT']
            skin_weights = np.load(skin_weight_path)['skin_weight']
            calib = self.get_calib(param)

            posed_v = linear_blend_skinning(mesh.vertices[None], skin_weights[None], joint_transform[None])
            projected_v = orthogonal(posed_v.transpose(1, 2), calib).transpose(1, 2)[0].numpy()
            projected_v[:, 1] *= -1
            posed_mesh = trimesh.Trimesh(projected_v, mesh.faces)

            return mesh, posed_mesh

    def load_normal(self, f_nml_path, b_nml_path):
        """
        Args:
            f_nml_path: path to normal
            b_nml_path: path to normal
        Returns:
            nmlF: FloatTensor [3, H, W]
        """
        nmlF = Image.open(f_nml_path).convert('RGB')
        nmlF = self.image_to_tensor(nmlF)

        nmlB = Image.open(b_nml_path).convert('RGB')
        nmlB = self.image_to_tensor(nmlB)

        return nmlF, nmlB

    def load_smpl(self, smpl_file):
        smpl_data = np.load(smpl_file)
        betas = torch.from_numpy(smpl_data['betas']).view(1, -1).float()
        poses = torch.from_numpy(smpl_data['pose']).view(1, 24, 3).float()
        calib = torch.from_numpy(smpl_data['calib']).view(1, 4, 4).float()

        with torch.no_grad():
            if self.cfg.smpl.model_type == 'smpl_vitruvian':
                smpl_model = smpl.create(**self.cfg.smpl)
                canon_smpl_v, canon_joints = smpl_model.initiate_vitruvian(
                    vitruvian_angle=self.cfg.smpl.vitruvian_angle,
                    custom_out=True,
                    betas=betas)

            else:
                smpl_model = self.smpl_model.cpu()
                smpl_out = smpl_model(betas=betas, custom_out=True)
                canon_smpl_v, canon_joints = smpl_out.vertices, smpl_out.joints

            posed_output = smpl_model(betas=betas, global_orient=poses[:, :1], body_pose=poses[:, 1:], custom_out=True)
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
        """
        Args:
            in_tensor: dict_keys(['sid', 'mask', 'betas', 'pose', 'scale', 'trans'])
            n_iter (int): number of iteration
            plot_gif (bool): save the optimization visual results as gif if true
        Returns:
            calib (torch.Tensor): [B, 4, 4]
            pose (torch.Tensor): [B, 24, 3]
            shape (torch.Tensor): [B, 10]
        """
        out_dir = in_tensor['out_dir']
        pose = in_tensor['pose'].clone().detach()  # [1, 23, 3]
        trans = in_tensor['trans'].clone().detach()  # [1, 3]
        betas = in_tensor['betas'].clone().detach().mean(0).unsqueeze(0)  # [1, 10]
        scales = in_tensor['scale']

        pose[:, [3, 6, 12, 15]] *= 0.2

        pose.requires_grad_()
        trans.requires_grad_()
        betas.requires_grad_()

        batch = len(scales)

        optimizer = torch.optim.SGD([pose, betas, trans], lr=1e-2, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=5)

        image = in_tensor['image'].to(self.device)
        mask = in_tensor['mask'].to(self.device)
        if self.normal_type == 'pifuhd':
            with torch.no_grad():
                nmlF, nmlB = self.NormalNet(image, mask)

        if plot_gif:
            filename_output = os.path.join(out_dir, 'smpl/optimize_smpl.gif')
            writer = imageio.get_writer(filename_output, mode='I', duration=0.05)

        loop_smpl = tqdm(range(n_iter))
        for i in loop_smpl:
            smpl_out = self.smpl_model(betas=betas, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=trans)
            smpl_verts = smpl_out.vertices * scales * torch.tensor([1.0, -1.0, -1.0]).view(1, 1, 3).to(self.device)

            visual_frames = []
            smpl_loss = 0
            for idx in range(batch):
                # render silhouette
                self.render.load_mesh(smpl_verts[idx], self.smpl_faces, use_normal=True)
                T_mask_F, T_mask_B = self.render.get_silhouette_image()
                T_normal_F, T_normal_B = self.render.get_clean_image()
                T_normal_F = T_normal_F.permute(0, 3, 1, 2) * 2 - 1
                T_normal_B = T_normal_B.permute(0, 3, 1, 2) * 2 - 1

                # silhouette loss
                diff_S = torch.abs(T_mask_F - mask[idx])

                # update icon normal
                if self.normal_type == 'icon':
                    with torch.no_grad():
                        nmlF, nmlB = self.NormalNet(image, mask, T_normal_F, T_normal_B)
                diff_F_smpl = torch.abs(T_normal_F - nmlF)
                diff_B_smpl = torch.abs(T_normal_B - nmlB)

                smpl_loss += diff_S.mean() + (diff_F_smpl.mean() + diff_B_smpl.mean()) * 0.5
                # smpl_loss += non_bg_loss

                visual_frames.append(diff_S[0].detach().cpu().numpy())

            loop_smpl.set_description(f"Body Fitting = {smpl_loss:.3f}")

            # save to gif file
            visual_frames = np.concatenate(visual_frames, 1)
            if plot_gif:
                writer.append_data(img_as_ubyte(visual_frames))

            # if i in [0, n_iter - 1]:
            #     cv2.imwrite(os.path.join(out_dir, 'iter' + str(i) + '.png'), visual_frames * 255)

            optimizer.zero_grad()
            smpl_loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step(smpl_loss)

            if i == n_iter - 1:
                im_name = in_tensor['name']
                # T_normal_F, T_normal_B = self.render.get_clean_image()
                # cv2.imwrite(f"{out_dir}/smpl/normalF/{im_name}.png", T_normal_F[0].detach().cpu().numpy()[..., ::-1] * 255)
                # cv2.imwrite(f"{out_dir}/normalB/{im_name}.png", T_normal_B[0].detach().cpu().numpy()[..., ::-1] * 255)

                nmlF = nmlF.permute(0, 2, 3, 1).detach().cpu().numpy() * 0.5 + 0.5
                nmlB = nmlB.permute(0, 2, 3, 1).detach().cpu().numpy() * 0.5 + 0.5
                cv2.imwrite(f"{out_dir}/normalF/{im_name}.png", nmlF[0, :, :, ::-1] * 255)
                cv2.imwrite(f"{out_dir}/normalB/{im_name}.png", nmlB[0, :, :, ::-1] * 255)

                save_obj_mesh('smpl.obj', smpl_verts[0].detach().cpu().numpy(), self.smpl_faces)

        calib = torch.stack(
            [torch.FloatTensor(
                [
                    [1, 0, 0, trans[0]],
                    [0, 1, 0, trans[1]],
                    [0, 0, -1, -trans[2]],
                    [0, 0, 0, 1],
                ]) * scale for trans, scale in zip(trans.cpu(), scales.cpu())]
        )
        betas = betas.detach().cpu()
        pose = pose.detach().cpu()
        # exit()
        return betas, pose, calib

    def generate_smpl(self, img_path, mask_path, save_dir):
        im_name = img_path.split('/')[-1][:-4]
        smpl_file = '%s/smpl/%s.npz' % (save_dir, im_name)

        image_hps = cv2.imread(img_path).astype(np.float32) / 255.
        image_hps = cv2.resize(image_hps, (224, 224))
        image_hps = torch.from_numpy(image_hps).permute(2, 0, 1)
        image_hps = self.image_to_pymaf_tensor(image_hps).unsqueeze(0)

        image_ori = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = self.mask_to_tensor(mask)
        image = self.image_to_tensor(image_ori)
        image *= mask.expand_as(mask)

        with torch.no_grad():
            preds_dict = self.hps(image_hps.to(self.device))
            output = preds_dict['smpl_out'][-1]
            cam, betas, pose = torch.split(output['theta'], [3, 10, 72], dim=1)
            scale, tranX, tranY = cam[0]
            trans = torch.tensor([tranX, tranY, 0.0]).view(1, 3).to(self.device)
        smpl_data = {
            'name': im_name,
            'image': image.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'betas': betas,
            'pose': pose,
            'scale': scale.view(1, -1),
            'trans': trans.view(1, -1),
            'out_dir': save_dir,
        }

        betas, poses, calib = self.optimize_smpl(smpl_data)
        np.savez(smpl_file, calib=calib.numpy(), betas=betas.numpy(), pose=poses.numpy())

    def get_item(self, index):
        img_path = self.image_files[index]
        items = img_path.split('/')
        im_name = items[-1][:-4]
        data_dir = os.path.dirname(os.path.dirname(img_path))

        mask_path = '%s/MASK/%s.png' % (data_dir, im_name)
        param_path = '%s/PARAM/%s.npz' % (data_dir, im_name)
        smpl_path = '%s/smpl/%s.npz' % (data_dir, im_name)
        icon_f_nml_path = '%s/normalF/%s.png' % (data_dir, im_name)
        icon_b_nml_path = '%s/normalB/%s.png' % (data_dir, im_name)

        if self.data_name == "rp":
            sid = data_dir.split('/')[-2]
            posed_obj_path = '%s/OBJ/mesh.obj' % data_dir
            canon_obj_path = '%s/%s/da_pose.obj' % (self.obj_dir, sid)
            res_path = './out/res/exp1-rp-normal-1view/rp/%s.obj' % sid
            data_dict = {
                'sid': sid,
                'im_name': im_name,
                'canon_obj_path': canon_obj_path,
                'posed_obj_path': posed_obj_path,
                'res_path': res_path,
                'param_path': param_path,
            }
            return data_dict

        elif self.data_name == "mvp":
            sid = data_dir.split('/')[-2]
            canon_obj_path = '%s/%s/da-pose.obj' % (self.obj_dir, sid)
            res_path = './out/res/exp1-mvp-normal-1view/mvp/%s.obj' % sid
            data_dict = {
                'sid': sid,
                'im_name': im_name,
                'canon_obj_path': canon_obj_path,
                'res_path': res_path,
                'skin_weight_path': os.path.join(self.obj_dir, sid, 'skin_weight.npz'),
                'param_path': param_path,
            }
            return data_dict

        if not os.path.exists(smpl_path):
            self.generate_smpl(img_path, mask_path, data_dir)

        image_ori = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask = self.mask_to_tensor(mask)
        image = self.image_to_tensor(image_ori)
        image = mask.expand_as(image) * image

        nmlF, nmlB = self.load_normal(icon_f_nml_path, icon_b_nml_path)
        # normal = torch.cat([nmlF, nmlB])

        data_dict = {
            'im_name': im_name,
            'b_min': self.b_min,
            'b_max': self.b_max,
            'rgb': image.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'normal': nmlF.unsqueeze(0),
            'smpl_lbs_weights': self.smpl_model.lbs_weights,  # [N, 24]
            'smpl_faces': torch.as_tensor(self.smpl_faces).long(),  # [6890, 3]
            'param_path': param_path,
        }

        if self.data_name == 'mvp':
            sid = data_dir.split('/')[-2]
            # res_dir = './out/res/exp3-rp/mvp/'
            data_dict.update({
                'sid': sid,
                'canon_obj_path': os.path.join(self.obj_dir, sid, 'da-pose.obj'),
                'skin_weight_path': os.path.join(self.obj_dir, sid, 'skin_weight.npz'),
            })
        elif self.data_name == 'buff':
            sid = data_dir.split('/')[-1]
            rid = int(im_name.split('_')[-1])
            obj_path = img_path.replace('RENDER', 'OBJ').replace('_' + str(rid), '')[:-4] + '.obj'
            res_path = img_path.replace('RENDER', 'results/exp3-rp-rgb')[:-4] + '.obj'
            assert os.path.exists(obj_path)
            data_dict.update({'sid': sid, 'rid': rid, 'posed_obj_path': obj_path, 'res_path': res_path})
        elif self.data_name == "rp":
            sid = data_dir.split('/')[-2]
            posed_obj_path = '%s/OBJ/mesh.obj' % data_dir
            canon_obj_path = '%s/%s/%s.obj' % (self.obj_dir, sid, sid)
            res_path = './out/res/arch++-normal-1view/rp/%s.obj' % sid
            data_dict.update({
                'sid': sid,
                'canon_obj_path': canon_obj_path,
                'posed_obj_path': posed_obj_path,
                'res_path': res_path
            })

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
