import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dream.provider import ViewDataset, MeshDataset
from dream.trainer import *
from dream.dlmesh import DLMesh
from dream.obj import Mesh
from dream.gui import GUI
from dream.sd import StableDiffusion
from dream.clip import CLIP

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, required=True, help="mesh template, must be obj format")
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_mesh', default=True, type=bool, help="export an obj mesh with texture")
    parser.add_argument('--stage', type=str, required=True, help="choose from [mesh, dmtet]")
    parser.add_argument('--train_face_ratio', type=float, default=0.3, help="training step ratio for face area")

    ### training options
    parser.add_argument('--tex_mlp', type=bool, default=False, help="optimize texture using albedo")
    parser.add_argument('--geo_mlp', type=bool, default=False, help="optimize texture using albedo")
    parser.add_argument('--skip_bg', action='store_true', help="using white background if skip background optimization")

    parser.add_argument('--lock_geo', action='store_true', help="fix geometry")
    parser.add_argument('--lock_tex', action='store_true', help="fix geometry")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--iters', type=int, default=5000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--warm_iters', type=int, default=500, help="training iters")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--albedo', action='store_true',
                        help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.,
                        help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--ssaa', type=int, default=2, help="super sampling anti-aliasing ratio")
    parser.add_argument('--w', type=int, default=512, help="render width in training")
    parser.add_argument('--h', type=int, default=512, help="render height in training")
    parser.add_argument('--anneal_tex_reso', action='store_true', help="increase h/w gradually for smoother texture")
    parser.add_argument('--init_empty_tex', action='store_true', help="always initialize an empty texture")

    parser.add_argument('--lambda_offsets', type=float, default=10, help="loss scale")
    parser.add_argument('--lambda_normal_offsets', type=float, default=10, help="loss scale")

    ### dataset options
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.2],
                        help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true',
                        help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60,
                        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=1.5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60,
                        help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### tetrahedral option
    parser.add_argument('--grid_res', type=int, default=256, help="tetrahedral grid resolution")
    parser.add_argument('--mesh_scale', type=float, default=1.1, help="tetrahedral mesh scale")
    parser.add_argument('--dmtet_iters', type=int, default=2000, help="demtet iters")
    parser.add_argument('--sdf_regularizer', type=float, default=0.2, help="sdf regularize loss")
    parser.add_argument('--laplace_scale', type=float, default=10000, help="laplace regularize scale")

    opt = parser.parse_args()

    # opt.fp16 = True # TODO: lead to mysterious NaNs in backward ???

    opt.dir_text = True

    if opt.albedo:
        opt.albedo_iters = opt.iters

    print(opt)

    seed_everything(opt.seed)


    model = DLMesh(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def build_dataloader(phase):
        """
        Args:
            phase: str one of ['train', 'test' 'val']
        Returns:
        """
        size = 4 if phase == 'val' else 100
        dataset = ViewDataset(opt, device=device, type=phase, size=size)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


    def configure_guidance():
        if opt.guidance == 'stable-diffusion':
            return StableDiffusion(device, opt.sd_version, opt.hf_key)
        else:
            return CLIP(device)


    def configure_optimizer():
        if opt.optim == 'adan':
            from optimizer import Adan

            optimizer = lambda model: Adan(
                model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(5 * opt.lr), betas=(0.9, 0.99), eps=1e-15)

        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x / opt.iters, 1))
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x * 0.0002, 1))
        return scheduler, optimizer


    if opt.test:
        trainer = Trainer(opt.stage, opt, model,
                          guidance=None,
                          device=device,
                          workspace=opt.workspace,
                          fp16=opt.fp16,
                          use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = GUI(opt, trainer)
            gui.render()

        else:
            test_loader = build_dataloader('test')

            trainer.test(test_loader)

            if opt.save_mesh:
                trainer.save_mesh()

    else:
        train_loader = build_dataloader('train')

        scheduler, optimizer = configure_optimizer()
        guidance = configure_guidance()
        trainer = Trainer(opt.stage,
                          opt=opt,
                          model=model,
                          guidance=guidance,
                          device=device,
                          workspace=opt.workspace,
                          optimizer=optimizer,
                          ema_decay=None,
                          fp16=opt.fp16,
                          lr_scheduler=scheduler,
                          use_checkpoint=opt.ckpt,
                          eval_interval=opt.eval_interval,
                          scheduler_update_every_step=True)

        if opt.gui:
            trainer.train_loader = train_loader  # attach dataloader to trainer
            gui = GUI(opt, trainer)
            gui.render()

        else:
            valid_loader = build_dataloader('val')
            max_epoch = np.ceil(opt.iters / (len(train_loader) * train_loader.batch_size)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # test
            test_loader = build_dataloader('test')
            trainer.test(test_loader)
            trainer.save_mesh()
