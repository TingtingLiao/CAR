import os
import glob
import sys
import cv2
import trimesh
import torch
import random
import imageio
import human_det
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from PIL import ImageFile
from skimage import img_as_ubyte
from glob import glob
import smpl
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from lib.pymaf.models import pymaf_net
from lib.pymaf.core import path_config
from lib.pymaf.utils.imutils import get_transformer
from lib.icon.model.NormalNet import get_normal_model
from lib.common.render import Render
from lib.common.train_util import concat_dict_tensor


class TestDataset():
    def __init__(self, cfg, image_dir, out_dir):
        random.seed(2022)
        self.cfg = cfg
        self.opt = cfg.dataset

        self.image_size = self.opt.input_size
        self.image_dir = image_dir
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.image_files = sorted(os.listdir(image_dir))
        self.num_views = self.opt.num_views
        self.b_min = np.array(self.opt.b_min)
        self.b_max = np.array(self.opt.b_max)

        self.device = torch.device(f'cuda:{cfg.training.gpus[0]}')

        self.render = Render(device=self.device)

        self.smpl_model = smpl.create(self.cfg.smpl.model_path,
                                      gender=self.cfg.smpl.gender,
                                      use_face_contour=self.cfg.smpl.use_face_contour).to(self.device)
        self.smpl_faces = self.smpl_model.faces.astype(np.int16)
        self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS).to(self.device)
        self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)['model'], strict=True)
        self.hps.eval()

        self.normal_type = 'icon'
        self.NormalNet = get_normal_model(self.normal_type).to(self.device)
        self.image_to_tensor, self.mask_to_tensor, self.image_to_pymaf_tensor = get_transformer(self.image_size)

        # self.generate_pifuhd_normal()

    def generate_pifuhd_normal(self):
        data_dir = "/media/liaotingting/usb3/Dataset/render_people64/synthetic"
        for sid in sorted(
                Path('/media/liaotingting/usb2/projects/ARWild/splits/rp.txt').read_text().strip().split('\n'))[
                   50:]:
            image, mask = self.load_render(os.path.join(data_dir, sid, "000/RENDER/180.png"))

            out_dir = f"/media/liaotingting/usb2/projects/ARWild/out/res/refine/rp/"
            os.makedirs(os.path.join(out_dir, "normalF"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "normalB"), exist_ok=True)

            with torch.no_grad():
                nmlF, nmlB = self.NormalNet(image[None].to(self.device), mask[None].to(self.device))
                nmlF = F.normalize(nmlF, eps=1e-6)
                nmlB = F.normalize(nmlB, eps=1e-6)

                nmlF[:, 0] *= -1
                nmlF[:, 2] *= -1

                nmlF = nmlF.permute(0, 2, 3, 1).detach().cpu().numpy() * 0.5 + 0.5
                nmlB = nmlB.permute(0, 2, 3, 1).detach().cpu().numpy() * 0.5 + 0.5

                nmlF = np.flip(nmlF, axis=2)
                # cv2.imwrite(f"{out_dir}/normalF/{sid}.png", nmlF[0, :, :, ::-1] * 255)
                cv2.imwrite(f"{out_dir}/normalB/{sid}.png", nmlF[0, :, :, ::-1] * 255)

    def __len__(self):
        return len(self.image_files) // self.num_views

    def load_normal(self, icon_f_nml_path, icon_b_nml_path, mask):
        """
        Args:
            icon_f_nml_path: str of subject id
            icon_b_nml_path: str of action id
            mask: FloatTensor [1, H, W]
        Returns:
            nmlF: FloatTensor [3, H, W]
        """
        nmlF = Image.open(icon_f_nml_path).convert('RGB')
        nmlF = self.image_to_tensor(nmlF)
        nmlF *= mask.expand_as(nmlF)

        nmlB = Image.open(icon_b_nml_path).convert('RGB')
        nmlB = self.image_to_tensor(nmlB)
        nmlB *= mask.expand_as(nmlB)

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

            posed_output = smpl_model(betas=betas, global_orient=poses[:, :3], body_pose=poses[:, 3:], custom_out=True)
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
        pose = in_tensor['pose'].clone().detach()  # [1, 23, 3]
        trans = in_tensor['trans'].clone().detach()  # [1, 3]
        betas = in_tensor['betas'].clone().detach().mean(0).unsqueeze(0)  # [1, 10]
        scales = in_tensor['scale']

        pose[:, [3, 6, 12, 15]] *= 0.5

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
            filename_output = os.path.join(self.out_dir, 'smpl/optimize_smpl.gif')
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
                cv2.imwrite(f"{self.out_dir}/normalF/{im_name}.png", nmlF[0, :, :, ::-1] * 255)
                cv2.imwrite(f"{self.out_dir}/normalB/{im_name}.png", nmlB[0, :, :, ::-1] * 255)

                smpl_mesh = trimesh.Trimesh(smpl_verts[0].detach().cpu().numpy(), self.smpl_faces,
                                            # {"process": False}
                                            )
                smpl_mesh.export(f"{self.out_dir}/normalB/{im_name}.obj")

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
        return betas, pose, calib

    def generate_smpl(self, img_path):
        im_name = img_path.split('/')[-1][:-4]

        image_hps = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        image_hps, mask = np.split(image_hps, [3], axis=2)
        image_hps *= np.repeat(mask, 3, axis=2)
        image_hps = cv2.resize(image_hps, (224, 224))
        image_hps = torch.from_numpy(image_hps).permute(2, 0, 1)
        image_hps = self.image_to_pymaf_tensor(image_hps).unsqueeze(0)

        image, mask = self.load_render(img_path)

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
            'trans': trans.view(1, -1)
        }
        betas, poses, calib = self.optimize_smpl(smpl_data)
        np.savez(f'{self.out_dir}/smpl/{im_name}.npz', calib=calib.numpy(), betas=betas.numpy(), pose=poses.numpy())

    # def process_images(self, save_crop_param=False):
    #     """
    #     remove remove background and crop and resize
    #     """
    #     import human_inst_seg
    #     import streamer_pytorch as streamer
    #     seg_engine = human_inst_seg.Segmentation()
    #     seg_engine.eval()
    #
    #     os.makedirs(self.mask_dir, exist_ok=True)
    #     image_files = sorted(glob.glob(f'{self.image_dir}/*'))
    #
    #     if save_crop_param:
    #         os.makedirs(os.path.join(self.out_dir, 'crop_param'), exist_ok=True)
    #
    #     data_stream = streamer.ImageListStreamer(image_files)
    #     loader = torch.utils.data.DataLoader(
    #         data_stream,
    #         batch_size=1,
    #         num_workers=1,
    #         pin_memory=False,
    #     )
    #     for data, im_path in tqdm(zip(loader, image_files)):
    #         outputs, bboxes, probs = seg_engine(data)
    #         bboxes = (bboxes * probs).sum(dim=1, keepdim=True) / probs.sum(dim=1, keepdim=True)
    #         bbox = bboxes[0, 0, 0].cpu().numpy().astype(np.int16)
    #         bbox = [bbox[1], bbox[3], bbox[0], bbox[2]]
    #
    #         image = (outputs[0, :3].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0
    #         mask = outputs[0, 3].cpu().numpy() * 255.0
    #         image, mask, bbox = crop(image, mask, bbox)
    #         cv2.imwrite(im_path, image[..., ::-1])
    #         cv2.imwrite(im_path.replace('images', 'masks'), mask)
    #         if save_crop_param:
    #             np.savetxt(im_path.replace('images', 'crop_param')[:-3] + 'txt', bbox)

    def prepare_data(self, file):
        # print(self.out_dir, self.image_dir, file)
        # exit()
        im_name = file[:-4]
        img_path = '%s/%s.png' % (self.image_dir, im_name)

        os.makedirs('%s/smpl' % self.out_dir, exist_ok=True)
        os.makedirs('%s/normalF' % self.out_dir, exist_ok=True)
        os.makedirs('%s/normalB' % self.out_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.out_dir, f'smpl/{im_name}.npz')):
            self.generate_smpl(img_path)

    def load_render(self, img_path):
        image = Image.open(img_path).convert('RGBA')
        mask = image.split()[-1]
        image = image.convert('RGB')

        mask = self.mask_to_tensor(mask)
        image = self.image_to_tensor(image)
        image *= mask.expand_as(image)
        return image, mask

    def get_item(self, index):
        # try:
        images = []
        masks = []
        normals = []
        smpl_dict = {}

        for idx in range(self.num_views):
            im_name = self.image_files[index + idx][:-4]
            # im_name = random.choice(self.image_files)[:-4]
            img_path = '%s/%s.png' % (self.image_dir, im_name)
            msk_path = img_path.replace("images", "masks")
            smpl_file = '%s/smpl/%s.npz' % (self.out_dir, im_name)
            icon_f_nml_path = '%s/normalF/%s.png' % (self.out_dir, im_name)
            icon_b_nml_path = '%s/normalB/%s.png' % (self.out_dir, im_name)

            self.prepare_data(self.image_files[index + idx])

            image, mask = self.load_render(img_path)

            smpl_dict = concat_dict_tensor(smpl_dict, self.load_smpl(smpl_file))

            nmlF, nmlB = self.load_normal(icon_f_nml_path, icon_b_nml_path, mask)
            normals.append(nmlF)

            images.append(image)
            masks.append(mask)

        # todo optimize batch with same shape
        smpl_dict['canon_smpl_vert'] = smpl_dict['canon_smpl_vert'][0]
        smpl_dict['canon_smpl_joints'] = smpl_dict['canon_smpl_joints'][0]

        data_dict = {
            'im_name': im_name,
            'b_min': self.b_min,
            'b_max': self.b_max,
            'rgb': torch.stack(images),
            'normal': torch.stack(normals),
            'mask': torch.stack(masks),
            'smpl_lbs_weights': self.smpl_model.lbs_weights,  # [N, 24]
            'smpl_faces': torch.as_tensor(self.smpl_faces).long(),  # [6890, 3]
        }

        data_dict.update(smpl_dict)

        return data_dict
        # except Exception as e:
        #     print(e)
        #     return self.get_item(random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)
