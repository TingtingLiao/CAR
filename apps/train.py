import sys
import os
import random
from tqdm import tqdm
import smpl
import pickle as pkl
import trimesh
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.data import TrainDataset, EvalDataset, Evaluator
from lib.model import HGPIFuNet
from lib.common.render import Render
from lib.common.checkpoints import CheckpointIO
from lib.common.train_util import *
from .refine import NormalRefine
torch.backends.cudnn.enabled = False


class Trainer(nn.Module):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.n_iter = 0
        self.star_epoch = 0
        self.cfg = cfg
        self.opt = cfg.training
        self.num_views = cfg.dataset.num_views
        self.max_epoch = self.opt.num_epoch
        self.model_name = f'{cfg.name}-{cfg.dataset.input_im}-{self.num_views}view'
        self.ckp_dir = os.path.join(self.opt.out_dir, 'ckpt', self.model_name)
        self.vis_dir = os.path.join(self.opt.out_dir, 'vis', self.model_name)
        self.log_dir = os.path.join(self.opt.out_dir, 'logs', self.model_name)
        self.res_dir = os.path.join(self.opt.out_dir, 'res', self.model_name)
        self.logger = SummaryWriter(self.log_dir)
        os.makedirs(self.ckp_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)

        self.device = torch.device(f"cuda:{self.opt.gpus[0]}")
        self.test_device = torch.device(f"cuda:{self.opt.test_gpus[0]}")
        self.netG = HGPIFuNet(cfg).to(self.device)
        self.optimizer_G = self.configure_optimizers()
        self.checkpoint_io = self.load_checkpoint()
        self.evaluator = Evaluator(self.test_device)
        self.render = Render(device=self.test_device)

    def load_checkpoint(self):
        checkpoint_io = CheckpointIO(self.ckp_dir, model=self.netG)
        load_dict = dict()
        if self.opt.resume:
            if os.path.exists(os.path.join(self.ckp_dir, 'model.pt')):
                load_dict = checkpoint_io.load('model.pt')
            elif os.path.exists(os.path.join(self.ckp_dir, 'latest.pt')):
                load_dict = checkpoint_io.load('latest.pt')
            elif os.path.exists(self.opt.resume_path):
                checkpoint_io.load(self.opt.resume_path)

        self.n_iter = load_dict.get('n_iter', 0)
        self.star_epoch = load_dict.get('epoch', 0)
        return checkpoint_io

    def _build_dataloader(self, dataset, mode):
        is_train = mode == "train"
        batch_size = self.opt.train_bsize if is_train else self.opt.val_bsize
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.opt.num_threads,
            pin_memory=True,
            drop_last=is_train,
        )

    def configure_optimizers(self):
        weight_decay = self.opt.weight_decay
        momentum = self.opt.momentum
        lr_G = self.opt.lr_G
        optim_params_G = [
            {
                'params': self.netG.parameters(),
                'lr': lr_G
            }
        ]
        if self.opt.optim == "Adadelta":
            optimizer_G = torch.optim.Adadelta(optim_params_G,
                                               lr=lr_G,
                                               weight_decay=weight_decay)
        elif self.opt.optim == "Adam":
            optimizer_G = torch.optim.Adam(optim_params_G,
                                           lr=lr_G)
        elif self.opt.optim == "RMSprop":
            optimizer_G = torch.optim.RMSprop(optim_params_G,
                                              lr=lr_G,
                                              weight_decay=weight_decay,
                                              momentum=momentum)
        else:
            raise NotImplementedError

        return optimizer_G

    def render_meshes(self, mesh_list, normalize):
        images = []
        for i, mesh in enumerate(mesh_list):
            self.render.load_mesh(mesh.vertices, mesh.faces, normalize=normalize)
            render_geo = self.render.get_clean_image(cam_ids=[0])
            render_geo = render_geo[0][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
            images.append(render_geo * 255)
        return images

    @staticmethod
    def convert_image_tensor_to_numpy(image_tensor):
        images = []
        for i, im in enumerate(image_tensor):
            im = np.uint8((im.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)[:, :, ::-1] * 255.0)
            images.append(im)
        return images

    @torch.no_grad()
    def visualize(self, data, save_path, return_posed=True, no_refine=True, refine_type='mesh_pose'):
        self.netG.eval()
        self.netG.training = False

        device = self.device
        pr_canon_mesh = gen_mesh(self.opt, self.netG, device, data, save_path)
        render = self.render_meshes([pr_canon_mesh], normalize=True)
        if return_posed:
            nn_lbs_weights = query_lbs_weight(pr_canon_mesh.vertices,
                                              data['canon_smpl_vert'],
                                              data['smpl_lbs_weights'],
                                              device).cpu()
            if no_refine:
                # warp directly
                _, projected_points = warp_and_project_points(pr_canon_mesh.vertices,
                                                              nn_lbs_weights,
                                                              data['joint_transform'],
                                                              data['calib'])
                projected_points = projected_points[0].numpy() * np.array([[1, -1, 1]])
                pr_posed_mesh = trimesh.Trimesh(projected_points, pr_canon_mesh.faces)
            else:
                refiner = NormalRefine(self.cfg.smpl, self.render, device)
                pr_posed_mesh = refiner.optimize_mesh(pr_canon_mesh, nn_lbs_weights, refine_type=refine_type, **data)
            render = self.render_meshes([pr_posed_mesh], normalize=False) + render

        inputs = self.convert_image_tensor_to_numpy(data['image'])
        vis_image = np.concatenate(inputs + render, 1)
        cv2.imwrite(save_path[:-4] + '.png', vis_image)
        print('visual image is saved to ', save_path[:-4] + '.png')
        if return_posed:
            return pr_canon_mesh, pr_posed_mesh
        else:
            return pr_canon_mesh

    def train_step(self, data):
        self.netG.train()

        in_tensor_dict = {
            k: v if k in ['labels', 'smpl_faces']
            else reshape_sample_tensor(v, self.num_views) if 'canon' in k
            else reshape_multiview_tensors(v)
            for k, v in data.items() if k in self.opt.input_keys
        }
        move_dict_to_device(in_tensor_dict, self.device)

        debug = False
        if debug:
            N = data['canon_surf_normal'].shape[-1]
            from lib.common.train_util import scatter_points_to_image
            img = scatter_points_to_image(data['canon_points'][0, :, N], data['image'][0])
            cv2.imwrite('%s/000-test.png' % self.vis_dir, img)
            exit()

        res, error, err_dict = self.netG.forward(in_tensor_dict)

        self.optimizer_G.zero_grad()
        error.backward()
        self.optimizer_G.step()

        return_dict = self.evaluator.calc_acc(res, in_tensor_dict['labels'], self.opt.thresh)
        return_dict.update(err_dict)

        for (k, v) in return_dict.items():
            self.logger.add_scalar("train_" + k, v, self.n_iter)

        return return_dict

    def eval_step(self, data):
        self.netG.eval()
        self.netG.training = False

        move_dict_to_batch(data)

        in_tensor_dict = {
            k: v if k in ['labels', 'smpl_faces']
            else reshape_sample_tensor(v, self.num_views) if 'canon' in k
            else reshape_multiview_tensors(v)
            for k, v in data.items() if k in self.opt.input_keys
        }
        move_dict_to_device(in_tensor_dict, self.device)

        res, error, err_dict = self.netG.forward(in_tensor_dict)

        return_dict = self.evaluator.calc_acc(res, in_tensor_dict['labels'], self.opt.thresh)
        for (k, v) in return_dict.items():
            self.logger.add_scalar("val_" + k, v, self.n_iter)

        return return_dict

    def run_train(self):
        # prepare data
        train_dataset = TrainDataset(self.cfg)
        train_data_loader = self._build_dataloader(train_dataset, 'train')
        print('train data size: ', len(train_data_loader))

        # NOTE: batch size should be 1 and use all the points for evaluation
        val_dataset = TrainDataset(self.cfg, phase='test')
        print('val data size: ', len(val_dataset))

        N = len(train_data_loader)
        star_epoch = self.star_epoch

        for i in range(10, len(val_dataset)):
            data = val_dataset.get_item(i)
            save_path = '%s/%s_iter%d_%s.obj' % (self.vis_dir, 'eval', self.n_iter, data['sid'])
            self.visualize(data, save_path)
        exit()

        for epoch in range(0, self.max_epoch):
            adjust_learning_rate(self.optimizer_G, epoch, self.opt.schedule, self.opt.gamma)
            if epoch < star_epoch:
                continue

            pbar = tqdm(enumerate(train_data_loader))
            for train_idx, data in pbar:
                out = self.train_step(data)

                string = f'{self.model_name} | {train_idx}/{N}| train | epoch:%d | lr:%s ' % (
                    epoch, self.optimizer_G.param_groups[0]['lr']) + convert_dict_to_str(out)
                pbar.set_description(string)

                if train_idx % int(self.opt.freq_save * N) == 0 and self.n_iter > 0:
                    self.checkpoint_io.save('model.pt', n_iter=self.n_iter, epoch=epoch)
                    self.checkpoint_io.save('latest.pt', n_iter=self.n_iter, epoch=epoch)

                if train_idx % int(self.opt.freq_show_train * N) == 0 and self.n_iter > 0:
                    data = random.choice(train_dataset)
                    save_path = '%s/%s_iter%d_%s.obj' % (self.vis_dir, 'train', self.n_iter, data['sid'])
                    self.visualize(data, save_path)

                if train_idx % int(self.opt.freq_show_val * N) == 0 and self.n_iter > 0:
                    data = random.choice(val_dataset)
                    save_path = '%s/%s_iter%d_%s.obj' % (self.vis_dir, 'eval', self.n_iter, data['sid'])
                    self.visualize(data, save_path)

                # if train_idx % int(self.opt.freq_eval * N) == 0 and self.n_iter > 0:
                #     metrics = {}
                #     for val_idx in tqdm(range(1)):
                #         data = val_dataset.get_item(random.randint(0, len(val_dataset)))
                #         out = self.eval_step(data)
                #         metrics = {k: metrics[k] + [v] if k in metrics else [v] for k, v in out.items()}
                #     metrics = {k: np.mean(np.array(v)) for k, v in metrics.items()}
                #     string = f' {self.model_name} | eval ' + convert_dict_to_str(metrics)
                #     pbar.set_description(string)
                #     f = open(f'{self.vis_dir}/metric_iter{self.n_iter}.pkl', 'wb')
                #     pkl.dump(metrics, f)
                #     f.close()

                self.n_iter += 1

    def test(self, dataset, save_error_map=False):
        """
        Args:
            dataset:
            save_error_map: save the error map if true
            spaces: only test metric in posed space
        Returns:
        """
        self.netG.eval()
        self.netG.training = False

        test_canon = dataset.data_name == 'mvp'

        metric_dict = {'posed_chamfer': [], 'posed_p2s': [], 'posed_nc': []}
        if test_canon:
            metric_dict.update({'canon_chamfer': [], 'canon_p2s': [], 'canon_nc': []})

        pbar = tqdm(range(len(dataset)))
        for i in pbar:
            data = dataset.get_item(i)
            save_path = f"{self.res_dir}/{dataset.data_name}/{data['sid']}_{data['im_name']}.obj"
            posed_save_path = f"{save_path[:-4]}_posed.obj"
            canon_metric_file = f'{save_path[:-4]}_metric_canon.npz'
            posed_metric_file = f'{save_path[:-4]}_metric_posed.npz'

            if os.path.exists(canon_metric_file):
                canon_metric = np.load(canon_metric_file)
            if os.path.exists(posed_metric_file):
                posed_metric = np.load(posed_metric_file)
            else:
                if os.path.exists(save_path) and os.path.exists(posed_save_path):
                    pr_canon_mesh = trimesh.load(save_path, **{'process': False})
                    pr_posed_mesh = trimesh.load(posed_save_path, **{'process': False})
                else:
                    pr_canon_mesh, pr_posed_mesh = self.visualize(data, save_path=save_path, return_posed=True)
                    pr_posed_mesh.export(posed_save_path)

                if test_canon:
                    gt_canon_mesh, gt_posed_mesh = dataset.load_gt_obj(**data)
                    self.evaluator.set_mesh(pr_canon_mesh, gt_canon_mesh)
                    canon_metric = self.evaluator.get_metrics()
                    np.savez(canon_metric_file, **canon_metric)
                    if save_error_map:
                        self.evaluator.get_error_map(save_path=f'{save_path[:-4]}_error_canon.png')
                else:
                    gt_posed_mesh = dataset.load_gt_obj(**data)

                self.evaluator.set_mesh(pr_posed_mesh, gt_posed_mesh)
                posed_metric = self.evaluator.get_metrics()
                np.savez(posed_metric_file, **posed_metric)
                if save_error_map:
                    self.evaluator.get_error_map(save_path=f'{save_path[:-4]}_error_posed.png')

            if test_canon:
                metric_dict.update({'canon_' + k: metric_dict['canon_'+k] + [v] for k, v in canon_metric.items()})
            metric_dict.update({'posed_' + k: metric_dict['posed_'+k] + [v] for k, v in posed_metric.items()})

            mean_dict = {k: np.array(v).mean() for k, v in metric_dict.items()}
            pbar.set_description(convert_dict_to_str(mean_dict))


if __name__ == '__main__':
    import argparse
    from lib.common.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str,
                        default="configs/ours-xyz.yaml",
                        help='Path to config file')
    parser.add_argument('-t', '--test', type=bool, default=False, help='debug mode')
    parser.add_argument('-r', '--resolution', type=int, default=256, help='resolution of marching cube')
    parser.add_argument('-nv', '--num_views', type=int, required=True, help='num views')
    parser.add_argument('-ii', '--input_image', type=str, default='normal', help='input image type [rgb or normal]')
    parser.add_argument('-re', '--resume', type=bool, default=True, help='resume model')
    parser.add_argument('-rp', '--resume_path', type=str, default="", help='resume model path')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='')
    args = parser.parse_args()

    cfg = load_config(args.config, 'configs/default.yaml')
    cfg.dataset.merge_from_list(['input_im', args.input_image, 'num_views', args.num_views])
    cfg.training.merge_from_list(['resume', args.resume,
                                  'gpus', [args.gpu],
                                  'mcube_res', args.resolution])
    cfg.freeze()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    trainer = Trainer(cfg)
    if args.test:
        test_dataset = EvalDataset('mvp', cfg, trainer.device)
        print('test data size: ', len(test_dataset))
        trainer.test(test_dataset)
    else:
        trainer.run_train()
