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

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from lib.pymaf.models import pymaf_net
from lib.pymaf.core import path_config
from lib.pymaf.utils.imutils import get_transformer
from lib.icon.model.NormalNet import get_icon_NormalNet
from lib.common.render import Render
from lib.common.train_util import concat_dict_tensor


def get_bbox(msk):
    rows = np.any(msk, axis=1)
    cols = np.any(msk, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin-100, cmax+100


def crop(img, msk, bbox=None):
    if bbox is None:
        bbox = get_bbox(msk > 100)
    cx = (bbox[3] + bbox[2]) // 2
    cy = (bbox[1] + bbox[0]) // 2

    w = img.shape[1]
    h = img.shape[0]
    height = int(1.138 * (bbox[1] - bbox[0]))
    hh = height // 2

    # crop
    if cy - hh < 0:
        img = cv2.copyMakeBorder(img, hh - cy, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        msk = cv2.copyMakeBorder(msk, hh - cy, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        cy = hh
    elif cy + hh > h:
        img = cv2.copyMakeBorder(img, 0, cy + hh - h, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        msk = cv2.copyMakeBorder(msk, 0, cy + hh - h, 0, 0, cv2.BORDER_CONSTANT, value=0)

    dw = min(cx, w - cx, hh)
    img = img[cy - hh:(cy + hh), cx - dw:cx + dw, :]
    msk = msk[cy - hh:(cy + hh), cx - dw:cx + dw]

    pts = np.array([cx - dw, cy - hh, cx + dw, cy + hh])

    dw = img.shape[0] - img.shape[1]
    if dw != 0:
        img = cv2.copyMakeBorder(img, 0, 0, dw // 2, dw // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        msk = cv2.copyMakeBorder(msk, 0, 0, dw // 2, dw // 2, cv2.BORDER_CONSTANT, value=0)

    img = cv2.resize(img, (512, 512))
    msk = cv2.resize(msk, (512, 512))

    # kernel = np.ones((1, 1), np.uint8)
    # msk = cv2.erode((255 * (msk > 100)).astype(np.uint8), kernel, iterations=1)

    return img, msk, pts


def uncrop(image, bbox, origin_shape, fill_val=127.5):
    h = bbox[3] - bbox[1]
    origin_im = cv2.resize(image, (h, h))
    origin_im = cv2.copyMakeBorder(origin_im,
                                   max(bbox[1], 0),
                                   max(512 - bbox[3], 0),
                                   max(bbox[0], 0),
                                   max(512 - bbox[2], 0), cv2.BORDER_CONSTANT, value=[fill_val, fill_val, fill_val])
    h, w = origin_shape

    origin_im = cv2.resize(origin_im, (h, h))
    crop_w = (h - w) // 2
    origin_im = origin_im[:, crop_w:-crop_w, :]

    return origin_im



class TestDataset():
    def __init__(self, cfg, data_dir):
        random.seed(2022)
        self.cfg = cfg
        self.opt = cfg.dataset
        self.image_size = self.opt.input_size
        self.data_dir = data_dir
        self.image_files = sorted(os.listdir(os.path.join(self.data_dir, 'images')))
        self.num_views = self.opt.num_views
        self.b_min = np.array(self.opt.b_min)
        self.b_max = np.array(self.opt.b_max)

        self.device = torch.device(f'cuda:{cfg.training.gpus[0]}')
        self.render = Render(device=self.device)
        self.smpl_model = smpl.create(cfg.smpl.path,
                                      gender=cfg.smpl.gender,
                                      use_face_contour=cfg.smpl.use_face_contour
                                      ).to(self.device)

        self.smpl_faces = self.smpl_model.faces.astype(np.int16)
        self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS).to(self.device)
        self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)['model'], strict=True)
        self.hps.eval()

        self.NormalNet = None

        self.image_to_tensor, self.mask_to_tensor, self.image_to_pymaf_tensor = get_transformer(self.image_size)

        # self.remove_background()

    def __len__(self):
        return len(self.image_files) // self.num_views

    def init_normal_net(self):
        self.NormalNet = get_icon_NormalNet()

    def load_normal(self, icon_f_nml_path, icon_b_nml_path):
        """
        Args:
            icon_f_nml_path: str of subject id
            icon_b_nml_path: str of action id
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
        cv2.imwrite(in_tensor['b_nml_save_path'],
                    (T_normal_B[0].detach().permute(1, 2, 0).cpu().numpy()[..., ::-1] * 0.5 + 0.5) * 255)
        cv2.imwrite(in_tensor['f_nml_save_path'],
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
                'f_nml_save_path': f'{save_dir}/normal_F_{im_name}.png',
                'b_nml_save_path': f'{save_dir}/normal_B_{im_name}.png'
            }
        betas, poses, calib = self.optimize_smpl(smpl_data)
        np.savez(f'{save_dir}/param_{im_name}.npz', calib=calib.numpy(), betas=betas.numpy(), pose=poses.numpy())

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

    def process_images(self, save_crop_param=False):
        """
        remove remove background and crop and resize
        """
        import human_inst_seg
        import streamer_pytorch as streamer
        seg_engine = human_inst_seg.Segmentation()
        seg_engine.eval()

        image_files = glob.glob(f'{self.data_dir}/images/*')
        os.makedirs(os.path.join(self.data_dir, 'masks'), exist_ok=True)
        if save_crop_param:
            os.makedirs(os.path.join(self.data_dir, 'crop_param'), exist_ok=True)

        data_stream = streamer.ImageListStreamer(image_files)
        loader = torch.utils.data.DataLoader(
            data_stream,
            batch_size=1,
            num_workers=1,
            pin_memory=False,
        )
        for data, im_path in tqdm(zip(loader, image_files)):
            outputs, bboxes, probs = seg_engine(data)
            bboxes = (bboxes * probs).sum(dim=1, keepdim=True) / probs.sum(dim=1, keepdim=True)
            bbox = bboxes[0, 0, 0].cpu().numpy().astype(np.int16)
            bbox = [bbox[1], bbox[3], bbox[0], bbox[2]]

            image = (outputs[0, :3].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0
            mask = outputs[0, 3].cpu().numpy() * 255.0
            image, mask, bbox = crop(image, mask, bbox)
            cv2.imwrite(im_path, image[..., ::-1])
            cv2.imwrite(im_path.replace('images', 'masks'), mask)
            if save_crop_param:
                np.savetxt(im_path.replace('images', 'crop_param')[:-3] + 'txt', bbox)

    def prepare_data(self, file):
        im_name = file[:-4]
        img_path = '%s/images/%s.png' % (self.data_dir, im_name)
        mask_path = '%s/masks/%s.png' % (self.data_dir, im_name)
        smpl_dir = '%s/smpl' % self.data_dir

        file_names = [f'param_{im_name}.npz', f'normal_B_{im_name}.png', f'normal_F_{im_name}.png']
        for file_name in file_names:
            if not os.path.exists(os.path.join(smpl_dir, file_name)):
                self.generate_smpl(img_path, mask_path, save_dir=smpl_dir)

        file_names = [f'normal_F_{im_name}_icon.png', f'normal_B_{im_name}_icon.png']
        for file_name in file_names:
            if not os.path.exists(os.path.join(smpl_dir, file_name)):
                self.generate_icon_normal(img_path, mask_path, smpl_dir)

    def get_item(self, index):
        # try:
        images = []
        masks = []
        normals = []
        smpl_dict = {}

        for id in range(self.num_views):
            im_name = self.image_files[index+id][:-4]
            # im_name = random.choice(self.image_files)[:-4]
            img_path = '%s/images/%s.png' % (self.data_dir, im_name)
            mask_path = '%s/masks/%s.png' % (self.data_dir, im_name)
            smpl_file = '%s/smpl/param_%s.npz' % (self.data_dir, im_name)
            icon_f_nml_path = '%s/smpl/normal_F_%s_icon.png' % (self.data_dir, im_name)
            icon_b_nml_path = '%s/smpl/normal_B_%s_icon.png' % (self.data_dir, im_name)

            self.prepare_data(self.image_files[index+id])

            image_ori = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            mask = self.mask_to_tensor(mask)
            image = self.image_to_tensor(image_ori)
            image = mask.expand_as(image) * image

            smpl_dict = concat_dict_tensor(smpl_dict, self.load_smpl(smpl_file))

            if self.opt.input_im == 'normal':
                nmlF, nmlB = self.load_normal(icon_f_nml_path, icon_b_nml_path)
                normal = torch.cat([nmlF, nmlB])
                image = nmlF
                normals.append(normal)

            images.append(image)
            masks.append(mask)

        # todo optimize batch with same shape
        smpl_dict['canon_smpl_vert'] = smpl_dict['canon_smpl_vert'][0]
        smpl_dict['canon_smpl_joints'] = smpl_dict['canon_smpl_joints'][0]

        data_dict = {
            'im_name': im_name,
            'b_min': self.b_min,
            'b_max': self.b_max,
            'image': torch.stack(images),
            'mask': torch.stack(masks),
            'smpl_lbs_weights': self.smpl_model.lbs_weights,  # [N, 24]
            'smpl_faces': torch.as_tensor(self.smpl_faces).long(),  # [6890, 3]
            'normal': torch.stack(normals)
        }

        data_dict.update(smpl_dict)

        return data_dict
        # except Exception as e:
        #     print(e)
        #     return self.get_item(random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)
