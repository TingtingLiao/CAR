import sys
import os
import random
from tqdm import tqdm
import trimesh
import torch
from torch.utils.tensorboard import SummaryWriter
from lib.common.render import Render

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.data import *
from lib.model import HGPIFuNet

from lib.common.train_util import *
from .basetrainer import BaseTrainer
from lib.common.lbs_util import query_lbs_weight, warp_and_project_points

torch.backends.cudnn.enabled = False


class Trainer(BaseTrainer):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.opt = cfg.training
        self.num_views = cfg.dataset.num_views
        self.max_epoch = self.opt.num_epoch
        self.model_name = f'{cfg.name}-{cfg.dataset.input_im}-{self.num_views}view'
        self.ckp_dir = os.path.join(self.opt.out_dir, 'ckpt', self.model_name)
        self.vis_dir = os.path.join(self.opt.out_dir, 'vis', self.model_name)
        self.log_dir = os.path.join(self.opt.out_dir, 'logs', self.model_name)
        self.logger = SummaryWriter(self.log_dir)
        os.makedirs(self.ckp_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.device = torch.device(f"cuda:{self.opt.gpus[0]}")
        self.test_device = torch.device(f"cuda:{self.opt.test_gpus[0]}")
        self.netG = HGPIFuNet(cfg).to(self.device)
        self.optimizer_G = self.configure_optimizers()
        self.checkpoint_io = self.load_checkpoint()
        self.evaluator = Evaluator(self.test_device)
        self.render = Render(device=self.test_device)

    def configure_optimizers(self):
        weight_decay = self.opt.weight_decay
        momentum = self.opt.momentum
        lr_G = self.opt.lr_G
        optim_params_G = [{'params': self.netG.parameters(), 'lr': lr_G}]
        if self.opt.optim == "Adadelta":
            optimizer_G = torch.optim.Adadelta(optim_params_G, lr=lr_G, weight_decay=weight_decay)
        elif self.opt.optim == "Adam":
            optimizer_G = torch.optim.Adam(optim_params_G, lr=lr_G)
        elif self.opt.optim == "RMSprop":
            optimizer_G = torch.optim.RMSprop(
                optim_params_G, lr=lr_G, weight_decay=weight_decay, momentum=momentum)
        else:
            raise NotImplementedError

        return optimizer_G

    @staticmethod
    def convert_image_tensor_to_numpy(image_tensor):
        images = []
        for i, im in enumerate(image_tensor):
            im = np.uint8((im.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)[:, :, ::-1] * 255.0)
            images.append(im)
        return images

    @torch.no_grad()
    def visualize(self, data, save_path, return_posed=True, no_refine=True, refine_type='mesh_pose', cam_ids=(0,)):
        self.netG.eval()
        self.netG.training = False
        device = self.device
        space = self.netG.space_list[0]
        # try:
        if space == 'pr':
            mesh = pifu_gen_mesh(self.opt, self.netG, device, data, save_path)
            render = self.render_meshes([mesh], normalize=False, cam_ids=cam_ids)
        else:
            mesh = gen_mesh(self.opt, self.netG, device, data, save_path)
            render = self.render_meshes([mesh], normalize=True, cam_ids=cam_ids)
            if return_posed:
                nn_lbs_weights = query_lbs_weight(mesh.vertices,
                                                  data['canon_smpl_vert'],
                                                  data['smpl_lbs_weights'],
                                                  device).cpu()

                # warp directly
                _, projected_points = warp_and_project_points(mesh.vertices,
                                                              nn_lbs_weights,
                                                              data['joint_transform'],
                                                              data['calib'])
                projected_points = projected_points[0].numpy() * np.array([[1, -1, 1]])
                pr_posed_mesh = trimesh.Trimesh(projected_points, mesh.faces)


                pr_posed_mesh.export(save_path[:-4] + '_posed.obj')
                render = self.render_meshes([pr_posed_mesh], normalize=False)

        inputs = self.convert_image_tensor_to_numpy(data['rgb'])
        vis_image = np.concatenate(inputs + render, 1)
        cv2.imwrite(save_path[:-4] + '.png', vis_image)
        print('visual image is saved to ', save_path[:-4] + '.png')
        if return_posed and space == 'canon':
            return mesh, pr_posed_mesh
        else:
            return mesh
        # except Exception as e:
        #     print(e)
        #     return

    def train_step(self, data):
        self.netG.train()
        in_tensor_dict = {
            k: v if k in ['labels', 'smpl_faces']
            else reshape_sample_tensor(v, self.num_views) if 'canon' in k or k == 'vT'
            else reshape_multiview_tensors(v)
            for k, v in data.items() if k in self.opt.input_keys
        }

        move_dict_to_device(in_tensor_dict, self.device)

        debug = False
        if debug:
            import torch.nn.functional as F
            from lib.common.geometry import index
            N = self.cfg.dataset.num_surface

            normal_posed = index(in_tensor_dict['image'], in_tensor_dict['projected_points'][:, :2])

            calib = data['calib'].to(self.device)
            vT = in_tensor_dict['vT']
            normal_canon = torch.bmm(torch.inverse(calib[:, :3, :3]), normal_posed)
            inverse_vT = torch.inverse(vT.reshape(-1, 4, 4)).view(vT.size(0), -1, 4, 4)
            normal_canon = torch.einsum('bvst,btv->bsv', inverse_vT[:, :, :3, :3], normal_canon)

            normal_canon = F.normalize(normal_canon)

            save_ply_mesh_with_color('a.ply', in_tensor_dict['canon_points'][0, :, :N].t().cpu().numpy(),
                                     normal_canon[0, :, :N].t().cpu().numpy() * 127.5 + 127.5)
            save_ply_mesh_with_color('b.ply', in_tensor_dict['canon_points'][0, :, :N].t().cpu().numpy(),
                                     in_tensor_dict['canon_surf_normal'][0].t().cpu().numpy() * 127.5 + 127.5)

            image = self.convert_image_tensor_to_numpy(in_tensor_dict['image'])
            cv2.imwrite('im.png', np.concatenate(image, 1))
            exit()
            from lib.common.train_util import scatter_points_to_image
            if not self.netG.sdf:
                inside_ids = in_tensor_dict['labels'][0, 0] > 0
                pts = in_tensor_dict['projected_points'][0, :, inside_ids]
            else:
                N = self.cfg.dataset.num_surface
                pts = in_tensor_dict['projected_points'][0, :, :N]

            img = scatter_points_to_image(pts, data['rgb'][0, 0])
            cv2.imwrite('%s/000-test.png' % self.vis_dir, img)

            pts[1] *= -1
            save_obj_mesh('pts.obj', pts.t().cpu().numpy())
            exit()

        res, error, err_dict = self.netG.forward(in_tensor_dict)

        self.optimizer_G.zero_grad()
        error.backward()
        self.optimizer_G.step()

        # save_samples_truncted_prob('/media/liaotingting/usb3/smpl.ply',
        #                            in_tensor_dict['projected_points'][0, :, self.netG.num_surf:].cpu().numpy().T,
        #                            res[0, :, self.netG.num_surf:].detach().cpu().numpy().T, thresh=0)

        if "labels" in in_tensor_dict:
            if self.netG.sdf:
                metric_dict = self.evaluator.calc_acc((res[:, :, self.netG.num_surf:]).float(),
                                                      in_tensor_dict['labels'][:, :, self.netG.num_surf:],
                                                      self.opt.thresh)
            else:
                metric_dict = self.evaluator.calc_acc(res, in_tensor_dict['labels'], self.opt.thresh)
            err_dict.update(metric_dict)

        for (k, v) in err_dict.items():
            self.logger.add_scalar("train_" + k, v, self.n_iter)

        return err_dict

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

    def get_dataset(self, phase):
        if self.opt.data_name == 'th':
            return THuman(self.cfg, phase)
        if self.opt.data_name == 'rp':
            return RenderPeople(self.cfg, phase)
        if self.opt.data_name == 'mvp':
            return MVPDataset(self.cfg, phase)
        else:
            raise ValueError('data_name must be one of [th, rp, mvp]')

    def run_train(self):
        # prepare data
        train_dataset = self.get_dataset('train')
        train_data_loader = self._build_dataloader(train_dataset, 'train')
        print('train data size: ', len(train_data_loader))

        # NOTE: batch size should be 1 and use all the points for evaluation
        val_dataset = self.get_dataset('test')
        print('val data size: ', len(val_dataset))

        N = len(train_data_loader)
        star_epoch = self.n_iter // N

        for epoch in range(0, self.max_epoch):
            lr = adjust_learning_rate(self.optimizer_G, epoch, self.opt.schedule, self.opt.gamma)
            if epoch < star_epoch:
                continue

            pbar = tqdm(train_data_loader)
            for data in pbar:
                out = self.train_step(data)

                print_str = f"{self.model_name} | it:{self.n_iter} | epoch:{epoch} | lr:{lr}"
                print_str += convert_dict_to_str(out)
                pbar.set_description(print_str)

                if self.n_iter % int(self.opt.freq_save * N) == 0 and self.n_iter > 0:
                    self.save_checkpoint(epoch)
                    self.save_checkpoint(epoch, 'latest.pt')

                if self.n_iter % int(self.opt.freq_show_train * N) == 0 and self.n_iter > 0:
                    data = random.choice(train_dataset)
                    save_path = '%s/%s_iter%d_%s.obj' % (self.vis_dir, 'train', self.n_iter, data['sid'])
                    self.visualize(data, save_path)

                if self.n_iter % int(self.opt.freq_show_val * N) == 0 and self.n_iter > 0:
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

            if epoch % self.opt.n_save_epoch == 0 and epoch > 0:
                self.save_checkpoint(epoch, f'epoch-{epoch:02}.pt')

    def test(self, dataset, save_error_map=False):
        """
        Args:
            dataset:
            save_error_map: save the error map if true
        Returns:
        """
        self.netG.eval()
        self.netG.training = False

        test_canon = dataset.data_name in ['mvp', 'rp']

        metric_dict = {'posed_chamfer': [], 'posed_p2s': [], 'posed_nc': []}
        if test_canon:
            metric_dict.update({'canon_chamfer': [], 'canon_p2s': [], 'canon_nc': []})

        pbar = tqdm(range(len(dataset)))
        for i in pbar:
            data = dataset.get_item(i)

            # if data['rid'] not in [0]:
            #     continue
            # save_path = f"{self.res_dir}/{dataset.data_name}/{data['sid']}_{data['im_name']}.obj"
            save_path = data['res_path']

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
                    pr_canon_mesh, pr_posed_mesh = self.visualize(data, save_path=save_path)

                if test_canon:
                    gt_canon_mesh, gt_posed_mesh = dataset.load_gt_obj(**data)
                    self.evaluator.set_mesh(pr_canon_mesh, gt_canon_mesh)
                    canon_metric = self.evaluator.get_metrics()
                    np.savez(canon_metric_file, **canon_metric)
                    if save_error_map:
                        self.evaluator.get_error_map(save_path=f'{save_path[:-4]}_error_canon.png')
                    # gt_posed_mesh.export('gt.obj')
                    # pr_posed_mesh.export('pr.obj')
                    # exit()

                else:
                    gt_posed_mesh = dataset.load_gt_obj(**data)

                self.evaluator.set_mesh(pr_posed_mesh, gt_posed_mesh)
                posed_metric = self.evaluator.get_metrics()
                np.savez(posed_metric_file, **posed_metric)
                if save_error_map:
                    self.evaluator.get_error_map(save_path=f'{save_path[:-4]}_error_posed.png')

            if test_canon:
                metric_dict.update({'canon_' + k: metric_dict['canon_' + k] + [v] for k, v in canon_metric.items()})
            metric_dict.update({'posed_' + k: metric_dict['posed_' + k] + [v] for k, v in posed_metric.items()})

            mean_dict = {k: np.array(v).mean() for k, v in metric_dict.items()}
            pbar.set_description(convert_dict_to_str(mean_dict))

            # exit()

    def eval(self):
        # NOTE: batch size should be 1 and use all the points for evaluation
        val_dataset = self.get_dataset('train')
        print('val data size: ', len(val_dataset))
        save_dir = os.path.join(self.opt.out_dir, 'res', self.model_name, self.opt.data_name)
        for i in range(len(val_dataset)):
            data = val_dataset.get_item(i)
            save_path = os.path.join(save_dir, data['sid'] + ".obj")
            if os.path.exists(save_path):
                self.visualize(data, save_path)
            exit()
