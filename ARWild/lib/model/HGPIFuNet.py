import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points, knn_gather
from termcolor import colored
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import *
from .HGFilters import *
from .net_util import init_net
from ..data.mesh_util import cal_sdf_batch


class HGPIFuNet(BasePIFuNet):
    def __init__(self, cfg, projection_mode='orthogonal'):
        super(HGPIFuNet, self).__init__(projection_mode=projection_mode)
        self.opt = cfg.net
        self.sdf = cfg.sdf
        self.use_filter = self.opt.use_filter
        self.input_im = cfg.dataset.input_im
        self.use_normal = self.opt.use_normal
        self.loss_type = self.opt.loss_type
        self.num_views = cfg.dataset.num_views
        self.num_surf = cfg.dataset.num_surface
        self.num_perturb = cfg.dataset.num_perturb
        self.num_bbox = cfg.dataset.num_bbox
        self.n_harmonic_functions = self.opt.n_harmonic_functions

        channels_IF = self.opt.mlp_dim
        # image encoder
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

        if self.use_normal:
            channels_IF[0] += 3
        channels_IF[0] += self.opt.point_feat_dim

        # geometric encoder on smpl body
        if 'pointnet2' in sum(self.opt.geo_feat_dict.values(), []):
            self.PointNetPlusPlus = PointNet2SSG()
        elif 'pointnet' in sum(self.opt.geo_feat_dict.values(), []):
            from .pointnet import PointNetEncoder, pointnet_kwarg
            self.PointNet = PointNetEncoder(**pointnet_kwarg)
        elif 'mlp' in sum(self.opt.geo_feat_dict.values(), []):
            self.point_encoder = ImplicitNet(
                dims=(3, 128, 128, 128, 128, 128),
                num_views=1,
                skip_in=(2, 3, 4),
                geometric_init=self.opt.geometric_init)

        init_net(self)

        self.space_list = []
        for space, val in self.opt.geo_feat_dict.items():
            if len(val) > 0:
                setattr(self, f"ImplicitNet_{space}", ImplicitNet(
                    dims=channels_IF,
                    num_views=self.num_views,
                    skip_in=self.opt.res_layers,
                    radius_init=self.opt.radius_init,
                    geometric_init=self.opt.geometric_init,
                    octaves_pe=self.opt.octaves_pe,
                    last_op=None if self.sdf else nn.Sigmoid()))
                setattr(self, f"pred_sdf_{space}", [])
                setattr(self, f"pred_normal_{space}", [])
                self.space_list.append(space)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.normal_im = None

        self.l1_loss = nn.SmoothL1Loss()
        self.l2_loss = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

    def filter(self, in_tensor_dict):
        """
        Parameters
        ----------
        in_tensor_dict (dict): input data

        Returns
        -------

        """
        if self.use_filter:
            self.im_feat_list, self.tmpx, self.normx = self.image_filter(in_tensor_dict[self.input_im])
            # If it is not in training, only produce the last im_feat
            if not self.training:
                self.im_feat_list = [self.im_feat_list[-1]]
            self.im_feat_list = [self.im_feat_list[-1]]
        else:
            self.im_feat_list = [in_tensor_dict[self.input_im]]

        if self.use_normal:
            self.normal_im = in_tensor_dict['normal']

    def _get_geo_feat(self, points, smpl_joints=None, smpl_vert=None, smpl_faces=None, feat_keys=None, **kwargs):
        """

        Parameters
        ----------
        points (torch.tensor): (B, 3, N)
        smpl_joints (torch.tensor): (B, 3, J) smpl joints
        smpl_vert (torch.tensor): (B, M, 3) smpl vertices and normal
        smpl_faces (torch.tensor): (B, M, 3) smpl vertices and normal
        feat_keys (list): [key1, key2, ...]
        kwargs

        Returns
        -------
        feats (torch.tensor): (B, D, N)

        """
        feat_list = []
        for feat_type in feat_keys:
            # points coordinate feature
            if feat_type == 'points':  # ARWild
                feat_list.append(points)

            elif feat_type == 'point_depth':  # PIFu
                feat_list.append(points[:, 2:3, :])

            # local spatial feature between samples and smpl
            elif feat_type == 'nndist':  # unsigned distance
                pts_dist, _, _ = knn_points(points.transpose(1, 2).contiguous(), smpl_vert)
                pts_dist = pts_dist.squeeze(2).unsqueeze(1)
                feat_list.append(pts_dist)

            elif feat_type == 'sdf':  # ICON
                smpl_sdf = cal_sdf_batch(smpl_vert, smpl_faces, points.transpose(1, 2).contiguous())
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

        return feat_list

    def get_geo_feat(self, space, canon_points=None, posed_points=None,
                     projected_points=None,
                     canon_smpl_joints=None, canon_smpl_vert=None,
                     posed_smpl_joints=None, posed_smpl_vert=None,
                     projected_smpl_joints=None, projected_smpl_vert=None,
                     smpl_faces=None, **kwargs):
        """
        Args:
            space (str): 'canon', 'posed' or 'projected'
            canon_points (torch.tensor): (B, 3, N)
            posed_points (torch.tensor): (B, 3, N)
            projected_points (torch.tensor): (B, 3, N)
            canon_smpl_joints (torch.tensor): (B, 3, N)
            posed_smpl_joints (torch.tensor): (B, 3, N)
            projected_smpl_joints:  (torch.tensor): (B, 3, N)
            canon_smpl_vert (torch.tensor): (B, N, 3)
            posed_smpl_vert (torch.tensor): (B, N, 3)
            projected_smpl_vert (torch.tensor): (B, N, 3)
            smpl_faces (torch.tensor): (N, 3)
        Returns:
            features (torch.tensor) (B, Dim, N)
        """
        feat_keys = self.opt.geo_feat_dict.get(space)

        if feat_keys == ["point_depth_joint"]:
            feats = torch.cat([posed_points[:, 2:3, :], canon_points[:, 2:3, :]], dim=1)
            return feats

        if feat_keys == ["joints"]:
            dist = (canon_points[:, :, None, :] - canon_smpl_joints[:, :, :, None])
            B, D, J, N = dist.size()
            dist = dist.view(B, D * J, N)
            return dist

        if space == 'canon':
            feats = self._get_geo_feat(canon_points,
                                       canon_smpl_joints,
                                       canon_smpl_vert,
                                       smpl_faces,
                                       feat_keys, **kwargs)
        elif space == 'posed':
            feats = self._get_geo_feat(posed_points,
                                       posed_smpl_joints,
                                       posed_smpl_vert,
                                       smpl_faces,
                                       feat_keys, **kwargs)
        elif space == 'projected':
            feats = self._get_geo_feat(projected_points,
                                       projected_smpl_joints,
                                       projected_smpl_vert,
                                       smpl_faces,
                                       feat_keys, **kwargs)
        else:
            raise ValueError('space should be canon, posed or projected')

        feats.requires_grad_()
        return feats

    @torch.enable_grad()
    def query(self, xyz, geo_feat, vT=None, calib=None, space='canon'):
        """
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        Parameters
        ----------
        xyz (torch.tensor): [B, 3, N] points in image space
        geo_feat (torch.tensor): [B, D, N] geometric features
        vT (torch.tensor): [B, N, 4, 4] world space to image space vertex transform matrix
        calib (torch.tensor): [B, 4, 4] camera calibration
        space (str)

        Returns
        -------

        """
        (xy, _) = xyz.split([2, 1], dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=False).detach().float()

        if self.num_views > 1:
            in_cube = in_cube.view(-1, self.num_views, in_cube.shape[-1])
            in_cube = in_cube.max(1)[0]

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        if not self.use_filter or self.use_normal:
            normal_posed = self.index(self.normal_im, xy)
            normal_canon = torch.bmm(torch.inverse(calib[:, :3, :3]), normal_posed)
            inverse_vT = torch.inverse(vT.reshape(-1, 4, 4)).view(vT.size(0), -1, 4, 4)
            normal_canon = torch.einsum('bvst,btv->bsv', inverse_vT[:, :, :3, :3], normal_canon)
            normal_canon = F.normalize(normal_canon)

            if not self.use_filter:
                feats = torch.cat([normal_canon, geo_feat], 1)
                if self.opt.skip_hourglass:
                    feats = torch.cat([tmpx_local_feature, feats], 1)
                pred = getattr(self, f"ImplicitNet_{space}")(feats)
                setattr(self, f"pred_sdf_{space}", [pred])
                return

        pred_list = []
        for im_feat in self.im_feat_list:
            if self.use_normal:
                feats = torch.cat([self.index(im_feat, xy), normal_canon, geo_feat], 1)
            else:
                feats = torch.cat([self.index(im_feat, xy), geo_feat], 1)

            if self.opt.skip_hourglass:
                feats = torch.cat([tmpx_local_feature, feats], 1)

            pred = getattr(self, f"ImplicitNet_{space}")(feats)
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

    def get_preds(self, space=None):
        if space is None:
            space = self.space_list[0]
        return getattr(self, f"pred_sdf_{space}")[-1]

    def _get_error(self, pred_sdf_list, gt_sdf=None, pred_normal_list=None, gt_surf_normal=None):

        # err_dict = {k: torch.tensor([0.], device=pred_sdf_list[0].device) for k in self.loss_type}
        err_dict = {k: 0 for k in self.loss_type}
        for i, pred_label in enumerate(pred_sdf_list):
            if 'occ' in self.loss_type:
                error_geo = self.l2_loss(pred_label, gt_sdf)
                err_dict['occ'] += error_geo * self.opt.lambda_occ

            elif 'sdf' in self.loss_type:
                error_geo = self.l1_loss(pred_label, gt_sdf)
                err_dict['sdf'] += error_geo * self.opt.lambda_sdf

            elif 'igr_sdf' in self.loss_type:
                sdf_surf, sdf_bbox = pred_label.split([self.num_surf, self.num_perturb + self.num_bbox], dim=2)
                error_surf_sdf = sdf_surf.abs().mean()
                err_dict['igr_sdf'] += self.opt.lambda_igr_surf_sdf * error_surf_sdf

                # if self.opt.lambda_igr_off_sdf > 0:
                #     occ = self.sigmoid(10 * sdf_bbox)
                #     gt_occ = torch.sign(gt_sdf) * 0.5 + 0.5
                #     error_off_surf_sdf = self.bce_loss(occ, gt_occ[:, :, self.num_surf:])
                #     err_dict['igr_sdf'] += self.opt.lambda_igr_off_sdf * error_off_surf_sdf

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

                error_off_surf_sdf = self.opt.lambda_igr_off_sdf * torch.exp(-10.0 * sdf_bbox.abs()).mean()
                err_dict['igr_sdf'] += error_off_surf_sdf

            if 'normal' in self.loss_type:
                nml_surf_pred = pred_normal_list[i][:, :, :self.num_surf]
                nml_surf_pred = F.normalize(nml_surf_pred)
                error_nml = (nml_surf_pred - gt_surf_normal).norm(2, dim=1).mean()
                err_dict['normal'] += self.opt.lambda_nml * error_nml

            if 'gradient' in self.loss_type:
                error_grad = (pred_normal_list[i][:, :, self.num_surf:].norm(2, dim=1) - 1).pow(2).mean()
                err_dict['gradient'] += error_grad * self.opt.lambda_grad

        err_dict = {k: v / len(pred_sdf_list) for k, v in err_dict.items()}

        return err_dict

    def calculate_error(self, in_tensor_dict):
        return_err_dict = {}
        gt_sdf = in_tensor_dict['labels'] if 'labels' in in_tensor_dict else None
        for space in self.space_list:
            pred_sdf_list = getattr(self, f"pred_sdf_{space}", None)
            pred_normal_list = getattr(self, f"pred_normal_{space}", None)
            gt_surf_normal = in_tensor_dict[
                f'{space}_surf_normal'] if f'{space}_surf_normal' in in_tensor_dict else None
            err_dict = self._get_error(pred_sdf_list,
                                       gt_sdf=gt_sdf, pred_normal_list=pred_normal_list, gt_surf_normal=gt_surf_normal)
            return_err_dict.update({space[0] + "/" + k: v for k, v in err_dict.items()})

        total_error = 0
        for k, v in return_err_dict.items():
            total_error += v
            return_err_dict[k] = v.item()
        return total_error, return_err_dict

    def get_normal(self, space, normalize=False):
        normal = getattr(self, f"pred_normal_{space}")[-1]
        if normalize:
            normal = F.normalize(normal, eps=1e-6)
        return normal

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
        self.filter(in_tensor_dict)
        for space in self.space_list:

            geo_feats = self.get_geo_feat(space=space, **in_tensor_dict)

            self.query(in_tensor_dict['projected_points'], geo_feats,
                       vT=in_tensor_dict['vT'] if 'vT' in in_tensor_dict else None,
                       calib=in_tensor_dict['calib'] if 'calib' in in_tensor_dict else None,
                       space=space)

            if self.sdf and self.training and ('normal' in self.loss_type or 'gradient' in self.loss_type):
                self.derive_normal(geo_feats, space=space)

        total_error, err_dict = self.calculate_error(in_tensor_dict)

        res = self.get_preds()

        return res, total_error, err_dict
