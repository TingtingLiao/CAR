import logging
import os
import torch
from torch.utils.data import DataLoader
from lib.common.checkpoints import CheckpointIO


class BaseTrainer(object):
    def __init__(self):
        self.n_iter = 0
        self.n_epoch = 0

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
        return checkpoint_io

    def save_checkpoint(self, n_epoch, ckp_name='model.pt'):
        self.checkpoint_io.save(ckp_name, n_iter=self.n_iter, n_epoch=n_epoch)

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

    def render_meshes(self, mesh_list, normalize, cam_ids=(0,)):
        images = []
        for i, mesh in enumerate(mesh_list):
            self.render.load_mesh(mesh.vertices, mesh.faces, normalize=normalize)
            render_geo = self.render.get_image(cam_ids=cam_ids)[..., ::-1]
            images.append(render_geo * 255)
        return images

    def evaluate(self, val_loader, val_dataset=None):
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError
