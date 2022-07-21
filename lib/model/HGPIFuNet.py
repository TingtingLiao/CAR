import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points, knn_gather
from termcolor import colored
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import *
from .HGFilters import *
from .transformer import ViT
from .net_util import init_net
from .geometry import orthogonal
from .pointnet2 import PointNet2SSG
from lib.data.mesh_util import cal_sdf_batch


class HGPIFuNet(BasePIFuNet):
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

    def __init__(self,
                 cfg,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss()):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)
        self.opt = cfg.net
        self.sdf = cfg.sdf
        self.use_filter = self.opt.use_filter
        self.use_attention = self.opt.use_attention
        self.loss_type = self.opt.loss_type
        self.num_views = cfg.dataset.num_views
        self.num_surf = cfg.dataset.num_surface
        self.num_perturb = cfg.dataset.num_perturb
        self.num_bbox = cfg.dataset.num_bbox

        self.n_harmonic_functions = self.opt.n_harmonic_functions
        channels_IF = self.opt.mlp_dim
        self.image_filter = None
        if self.opt.no_im_feat:
            channels_IF[0] = 0
        elif self.use_filter:
            if self.opt.gtype == "HGPIFuNet":
                self.image_filter = HGFilter(self.opt)
            else:
                print(colored(f"Backbone {self.opt.gtype} is unimplemented", 'green'))

            channels_IF[0] = self.opt.hourglass_dim
        else:
            channels_IF[0] = 3
        channels_IF[0] += self.opt.point_feat_dim

        # if 'pointnet2' in [item[1] for _, feat_keys in self.opt.geo_feat]:
        #     self.PointNetPlusPlus = PointNet2SSG()
        # elif 'pointnet' in [item[1] for item in self.opt.geo_feat]:
        #     from .pointnet import PointNetEncoder, pointnet_kwarg
        #     self.PointNet = PointNetEncoder(**pointnet_kwarg)
        # elif 'mlp' in [item[1] for item in self.opt.geo_feat]:
        #     self.point_encoder = ImplicitNet(
        #         dims=(3, 128, 128, 128, 128, 128),
        #         num_views=1,
        #         skip_in=(2, 3, 4),
        #         geometric_init=self.opt.geometric_init)

        init_net(self)

        for space in self.opt.geo_feat_dict:
            setattr(self, f"ImplicitNet_{space}", ImplicitNet(dims=channels_IF,
                                                              num_views=self.num_views,
                                                              skip_in=self.opt.res_layers,
                                                              geometric_init=self.opt.geometric_init,
                                                              last_op=None if self.sdf else nn.Sigmoid()))
            setattr(self, f"pred_sdf_{space}", [])
            setattr(self, f"pred_normal_{space}", [])

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.l1_loss = nn.SmoothL1Loss()
        self.l2_loss = nn.MSELoss()

    def filter(self, images):
        '''
        Filter the input images store all intermediate features.
        Args:
            images (torch.tensor): [B, C, H, W] input images
        '''
        if self.use_filter:
            self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        else:
            self.im_feat_list = [images]
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def _get_geo_feat(self, points, smpl_joints=None, smpl_vert=None, smpl_faces=None, feat_keys=None):
        """
        Args:
            points (torch.tensor): (B, 3, N)
            smpl_joints (torch.tensor): (B, 3, J) smpl joints
            smpl_vert (torch.tensor): (B, M, 3) smpl vertices and normal
            smpl_faces (torch.tensor): (B, M, 3) smpl vertices and normal
        Returns:
            feats (torch.tensor): (B, D, N)
        """
        feat_list = []
        for feat_type in feat_keys:
            # points coordinate feature
            if feat_type == 'points':  # ARWild
                feat_list.append(points)

            elif feat_type == 'depth':  # PIFu
                feat_list.append(points[:, 2:3, :])

            # local spatial feature between samples and smpl
            elif feat_type == 'nndist':  # unsigned distance
                dist, _, _ = knn_points(points.transpose(1, 2).contiguous(), smpl_vert)
                dist = dist.squeeze(2).unsqueeze(1)
                feat_list.append(dist)

            elif feat_type == 'sdf':  # ICON
                smpl_sdf, smpl_norm = cal_sdf_batch(smpl_vert, smpl_faces, points.permute(0, 2, 1).contiguous())
                feat_list.append(smpl_sdf.transpose(1, 2))

            elif feat_type == 'joints':  # ARCH
                dist = (points[:, :, None, :] - smpl_joints[:, :, :, None]) ** 2
                B, D, J, N = dist.size()
                feat = torch.exp(-dist.contiguous().view(B, D * J, N))
                feat_list.append(feat)

            # global encoder for smpl
            elif feat_type == 'mlp':  # mlp encoder
                feat = self.point_encoder(smpl_vert.transpose(1, 2))
                dist, ids, _ = knn_points(points.transpose(1, 2).contiguous(), smpl_vert)
                feat = knn_gather(feat.transpose(1, 2), ids)  # (B, N, K, Dim)
                feat_list.append(feat.squeeze(2).transpose(1, 2))
            elif feat_type == 'pointnet2':  # ARCH++
                feat_list.append(self.PointNetPlusPlus(smpl_vert, points))
            elif feat_type == 'pointnet':
                feat_list.append(self.PointNet(smpl_vert.transpose(1, 2), points))
            else:
                raise NotImplementedError()

        feat_list = torch.cat(feat_list, 1)
        if self.sdf and self.training:
            feat_list.requires_grad_()

        return feat_list

    def get_geo_feat(self, space, canon_points, posed_points, projected_points=None,
                     canon_smpl_joints=None, posed_smpl_joints=None,
                     canon_smpl_vert=None, posed_smpl_vert=None, smpl_faces=None, **kwargs):
        """
        Args:
            space (str): 'canon', 'posed' or 'projected'
            canon_points (torch.tensor): (B, 3, N)
            posed_points (torch.tensor): (B, 3, N)
            projected_points (torch.tensor): (B, 3, N)
            canon_smpl_joints (torch.tensor): (B, 3, N)
            posed_smpl_joints (torch.tensor): (B, 3, N)
            canon_smpl_vert (torch.tensor): (B, N, 3)
            posed_smpl_vert (torch.tensor): (B, N, 3)
            smpl_faces (torch.tensor): (N, 3)
        Returns:
            features (torch.tensor) (B, Dim, N)
        """
        feat_keys = self.opt.geo_feat_dict.get(space)
        if space == 'canon':
            feats = self._get_geo_feat(canon_points,
                                       canon_smpl_joints,
                                       canon_smpl_vert,
                                       smpl_faces,
                                       feat_keys)

        elif space == 'posed':
            feats = self._get_geo_feat(posed_points,
                                       posed_smpl_joints,
                                       posed_smpl_vert,
                                       smpl_faces,
                                       feat_keys)
        elif space == 'projected':
            # only used for PIFu, using z value in projected space
            feats = self._get_geo_feat(projected_points, feat_keys=feat_keys)
        else:
            raise ValueError('space should be canon, posed or projected')

        return feats

    @torch.enable_grad()
    def query(self, xyz, geo_feat, space='canon'):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param xyz: [B, 3, N] points in image space
        :param geo_feat: [B, dim, N] geometric features
        :param space: str
        :return: [B, Res, N] predictions for each point
        '''
        (xy, _) = xyz.split([2, 1], dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=False).detach().float()

        if self.num_views > 1:
            in_cube = in_cube.view(-1, self.num_views, in_cube.shape[-1])
            in_cube = in_cube.max(1)[0]

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        pred_list = []
        for im_feat in self.im_feat_list:
            # for space, geo_feat in geo_feats_dict.items():
            features = torch.cat([self.index(im_feat, xy), geo_feat], 1)
            if self.opt.skip_hourglass:
                features = torch.cat([tmpx_local_feature, features], 1)

            pred = getattr(self, f"ImplicitNet_{space}")(features)
            if not self.sdf:  # out of image plane is always set to 0
                pred = pred * in_cube[:, None]

            pred_list.append(pred)
        setattr(self, f"pred_sdf_{space}", pred_list)

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_preds(self, space='canon'):
        return getattr(self, f"pred_sdf_{space}")[-1]

    def _get_error(self, pred_sdf_list, gt_sdf=None, pred_normal_list=None, gt_surf_normal=None):
        err_dict = {k: 0 for k in self.loss_type}

        for i, pred_label in enumerate(pred_sdf_list):
            if 'occ' in self.loss_type:
                error_geo = self.l2_loss(pred_label, gt_sdf)
                err_dict['occ'] += error_geo * self.opt.lambda_occ

            elif 'sdf' in self.loss_type:
                error_geo = self.l1_loss(pred_label, gt_sdf)
                err_dict['sdf'] += error_geo * self.opt.lambda_sdf

            elif 'igr_sdf' in self.loss_type:
                sdf_surf, sdf_off_suf = pred_label.split([self.num_surf, self.num_perturb + self.num_bbox], dim=2)
                error_surf_sdf = self.opt.lambda_igr_surf_sdf * sdf_surf.abs().mean()

                pred_sign = (sdf_off_suf > 0).float()
                gt_sign = (gt_sdf[:, :, self.num_surf:] > 0).float()
                error_off_surf_sdf = self.opt.lambda_igr_off_sdf * self.l2_loss(pred_sign, gt_sign)

                # weighted bbox sdf loss
                # bbox_points = in_tensor_dict['canon_points'][:, :, -self.num_bbox:]
                # smpl_vert = in_tensor_dict['canon_smpl_vert']
                # with torch.no_grad():
                #     # compute the distance from point to smpl surface
                #     dist, _, _ = knn_points(bbox_points.transpose(1, 2).contiguous(), smpl_vert)  # (B, N, 1)
                #     # the far, the weights are larger
                #     weights = F.normalize(dist.squeeze(2))  # (B, N)
                # error_off_surf_sdf = self.opt.lambda_igr_off_sdf * \
                #                      (torch.exp(-sdf_bbox.squeeze(1).abs()) * weights).sum(1).mean()

                # SCANimate
                # error_off_surf_sdf = self.opt.lambda_igr_off_sdf * torch.exp(-10.0 * sdf_bbox.abs()).mean()
                err_dict['igr_sdf'] += error_surf_sdf + error_off_surf_sdf

            if 'normal' in self.loss_type:
                nml_surf_pred = pred_normal_list[i][:, :, :self.num_surf]
                canon_nml_surf_gt = gt_surf_normal
                error_nml = (nml_surf_pred - canon_nml_surf_gt).norm(2, dim=1).mean()
                err_dict['normal'] += self.opt.lambda_nml * error_nml

            if 'gradient' in self.loss_type:
                error_grad = (pred_normal_list[i][:, :, self.num_surf:].norm(2, dim=1) - 1).pow(2).mean()
                err_dict['gradient'] += error_grad * self.opt.lambda_grad

        err_dict = {k: v / len(pred_sdf_list) for k, v in err_dict.items()}

        return err_dict

    def calculate_error(self, in_tensor_dict):
        return_err_dict = {}
        gt_sdf = in_tensor_dict['labels'] if 'labels' in in_tensor_dict else None
        for space in self.opt.geo_feat_dict.keys():
            pred_sdf_list = getattr(self, f"pred_sdf_{space}")
            pred_normal_list = getattr(self, f"pred_normal_{space}")
            err_dict = self._get_error(pred_sdf_list,
                                       gt_sdf=gt_sdf,
                                       pred_normal_list=pred_normal_list,
                                       gt_surf_normal=in_tensor_dict[f'{space}_surf_normal'])
            return_err_dict.update({space + k: v for k, v in err_dict.items()})

        total_error = 0
        for k, v in return_err_dict.items():
            total_error += v
            return_err_dict[k] = v.item()
        return total_error, return_err_dict

    def derive_normal(self, inputs, space='canon', normalize=False):
        pred_list = getattr(self, f"pred_sdf_{space}")
        normal_list = []
        for pred in pred_list:
            grad = gradient(inputs, pred)[:, -3:, :]
            if normalize:
                grad = F.normalize(grad, eps=1e-6)
            normal_list.append(grad)
        setattr(self, f"pred_normal_{space}", normal_list)

    def forward(self, in_tensor_dict):
        self.filter(in_tensor_dict['image'])

        for space in self.opt.geo_feat_dict.keys():
            geo_feats = self.get_geo_feat(space=space, **in_tensor_dict)

            self.query(in_tensor_dict['projected_points'], geo_feats, space=space)

            if 'normal' in self.loss_type and self.training:
                self.derive_normal(geo_feats, space=space)

        total_error, err_dict = self.calculate_error(in_tensor_dict)

        res = self.get_preds()

        return res, total_error, err_dict
