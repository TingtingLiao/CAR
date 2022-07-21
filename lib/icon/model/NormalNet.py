# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from .FBNet import define_G
from lib.model.BasePIFuNet import BasePIFuNet
from lib.icon.configs.config import cfg


class NormalNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self, cfg, error_term=nn.SmoothL1Loss()):

        super(NormalNet, self).__init__(error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()

        self.opt = cfg.net

        self.in_nmlF = [
            item[0] for item in self.opt.in_nml
            if '_F' in item[0] or item[0] == 'image'
        ]
        self.in_nmlB = [
            item[0] for item in self.opt.in_nml
            if '_B' in item[0] or item[0] == 'image'
        ]
        self.in_nmlF_dim = sum([
            item[1] for item in self.opt.in_nml
            if '_F' in item[0] or item[0] == 'image'
        ])
        self.in_nmlB_dim = sum([
            item[1] for item in self.opt.in_nml
            if '_B' in item[0] or item[0] == 'image'
        ])

        self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(self.in_nmlB_dim, 3, 64, "global", 4, 9, 1, 3, "instance")

    def forward(self, in_tensor):
        """
        Args:
            in_tensor:
             'image': [B, 3, 512, 512],
            'T_normal_F': [B, 3, 512, 512],
            'T_normal_B': [B, 3, 512, 512],

        Returns:
        """
        inF_list = []
        inB_list = []

        for name in self.in_nmlF:
            inF_list.append(in_tensor[name])
        for name in self.in_nmlB:
            inB_list.append(in_tensor[name])
        nmlF = self.netF(torch.cat(inF_list, dim=1))
        nmlB = self.netB(torch.cat(inB_list, dim=1))

        nmlF = nmlF * in_tensor['mask']
        nmlB = nmlB * in_tensor['mask']

        return nmlF, nmlB


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


def get_icon_NormalNet():
    cfg.merge_from_file('lib/icon/configs/icon-filter.yaml')
    model = NormalNet(cfg)

    assert os.path.exists(cfg.normal_path)

    model_dict = model.state_dict()
    normal_dict = torch.load(cfg.normal_path)['state_dict']

    for key in normal_dict.keys():
        normal_dict = rename(normal_dict, key,
                             key.replace("netG.", ""))

    normal_dict = {
        k: v
        for k, v in normal_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    print(f"Resume normal model from {cfg.normal_path}")

    model_dict.update(normal_dict)
    model.load_state_dict(model_dict)

    model.training = False
    model.eval()

    del normal_dict
    del model_dict

    return model
