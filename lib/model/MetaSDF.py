from collections import OrderedDict
import torch
from torchmeta.utils.gradient_based import gradient_update_parameters
from .BaseSDFNet import BaseSDFNet
from .hyperlayers import HyperNet
from .sample import Sampler
from .net_util import MetaMLP
from lib.common.train_util import get_mgrid


class SmplSDFNet(BaseSDFNet):
    def __init__(self, cfg):
        super(SmplSDFNet, self).__init__(cfg)
        self.condition = cfg.training.condition
        self.hyper_net = HyperNet(encoder=MetaMLP(**cfg.net.smpl_encoder_kwargs),
                                  sdf_net=self.sdf_net, **cfg.net.smpl_hypernet_kwargs)
        self.sampler = Sampler(local_sigma=cfg.dataset.sigma)

    def get_sdf_net_param(self, smpl_v, hyper_params=None):
        return self.hyper_net(smpl_v, hyper_params)

    @torch.enable_grad()
    def inner_loop(self, surf_points, surf_normal, latent_input, num_meta_steps=2, alpha=0.01):
        meta_batch = surf_points.shape[0]
        hyper_params = OrderedDict(self.hyper_net.meta_named_parameters())
        for k, v in hyper_params.items():
            hyper_params[k] = v.unsqueeze(0).repeat((meta_batch,) + (1,) * len(v.shape))

        for _ in range(num_meta_steps):
            sdf_params = self.get_sdf_net_param(latent_input, hyper_params)
            loss, loss_dict = self.forward_with_param(surf_points, surf_normal, sdf_params)
            self.hyper_net.zero_grad()
            hyper_params = gradient_update_parameters(self.hyper_net, loss,
                                                      params=hyper_params,
                                                      step_size=alpha,
                                                      first_order=True)
        return hyper_params

    def forward_with_param(self, points, surf_normal, sdf_params=None):
        points.requires_grad_()
        predictions = self.sdf_net(points, params=sdf_params)
        pred_normal = self.derive_normal(points, predictions)
        return self.get_error(predictions, pred_normal, surf_normal)

    def get_data(self, in_tensor_dict):
        if self.condition == 'smpl':
            smpl_vert = in_tensor_dict['smpl_vert']
            smpl_normal = in_tensor_dict['smpl_normal']
            condition_input = torch.cat([smpl_vert, smpl_normal], -1)
            points = self.sampler.get_points(smpl_vert)
            return points, smpl_normal, condition_input
        elif self.condition == 'depth_pcl':
            samples = in_tensor_dict['samples']
            surf_normal = in_tensor_dict['surf_normal']
            condition_input = in_tensor_dict['depth_pcl']
            return samples, surf_normal, condition_input
        else:
            raise ValueError()

    def forward(self, in_tensor_dict):
        samples, surf_normal, condition_input = self.get_data(in_tensor_dict)
        sdf_params = self.get_sdf_net_param(condition_input)
        outer_loss, loss_dict = self.forward_with_param(samples, surf_normal, sdf_params)

        return outer_loss, loss_dict


