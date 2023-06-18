from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from .pointnet2_utils import QueryAndGroup, GroupAll, sample_farthest_points, grouping_operation


arch_plusplus_kwarg = {
    'sa_param': [
        {'npoint': 1024, 'radius': 0.1, 'nsample': 16, 'mlp': [3, 16, 32]},
        {'npoint': 512, 'radius': 0.2, 'nsample': 32, 'mlp': [32, 32, 64]},
        {'npoint': 128, 'radius': 0.4, 'nsample': 64, 'mlp': [64, 64, 128]}
    ],
    'fp_param': [
        {'mlp': [32+3, 32, 32]},
        {'mlp': [64+3, 32, 32]},
        {'mlp': [128+3, 32, 32]}
    ]
}


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
            self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """
        new_features_list = []
        idx = sample_farthest_points(xyz, self.npoint)
        new_xyz = grouping_operation(xyz, idx)

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features) if features is not None else self.groupers[i](
                xyz, new_xyz)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)

            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        B, N, C = unknown.shape
        if known is not None:
            dist, idx, _ = knn_points(unknown, known, K=3)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = torch.sum(
                grouping_operation(known_feats.transpose(1, 2), idx) * weight.view(B, N, 3, 1), dim=2)
            interpolated_feats = interpolated_feats.transpose(1, 2)

        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )
        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class PointNet2SSG(nn.Module):
    def __init__(self, use_xyz=True):
        super().__init__()

        self.use_xyz = use_xyz
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.FP_modules = nn.ModuleList()

        for kwarg in arch_plusplus_kwarg['sa_param']:
            self.SA_modules.append(PointnetSAModule(use_xyz=self.use_xyz, **kwarg))
        for kwarg in arch_plusplus_kwarg['fp_param']:
            self.FP_modules.append(PointnetFPModule(**kwarg))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 0:3].transpose(1, 2).contiguous()

        return xyz, features

    @staticmethod
    def normalize_pc(surface_pcl, query_pcl):
        """
        Args:
            surface_pcl (torch.FloatTensor):  [B, N, 3]
            query_pcl (torch.FloatTensor): [B, N, 3]

        Returns:
            surface_pcl_normalized: [B, N, 3]
            query_pcl_normalized: [B, N, 3]
        """
        vmax = surface_pcl.max(1)[0]
        vmin = surface_pcl.min(1)[0]
        center = (vmax + vmin) * 0.5
        scale = (vmax[:, 1] - vmin[:, 1]) * 0.5

        surface_pcl_normalized = (surface_pcl - center[:, None, :]) / scale[:, None, None]
        query_pcl_normalized = (query_pcl - center[:, None, :]) / scale[:, None, None]
        return surface_pcl_normalized, query_pcl_normalized

    # def forward_(self, surface_pcl, query_pcl):
    #     """
    #     Args:
    #         surface_pcl (torch.FloatTensor): [B, N, 6]
    #         query_pcl: (torch.FloatTensor)[B, 3, N]
    #     Returns:
    #         feat (torch.FloatTensor): [B, D, N]
    #     """
    #     xyz, query_pcl_normalized = self.normalize_pc(surface_pcl, query_pcl.transpose(1, 2))
    #     features = xyz.transpose(1, 2)
    #     feat_list = []
    #     for sa_module, fp_module in zip(self.SA_modules, self.FP_modules):
    #         xyz, features = sa_module(xyz, features)
    #         print(xyz.shape, features.shape)
    #         exit()
    #
    #         from lib.data.mesh_util import save_obj_mesh
    #         save_obj_mesh('/media/liaotingting/usb3/mesh.obj', xyz[0].view(-1, 3).cpu().numpy())
    #         exit()
    #
    #         feat = fp_module(query_pcl_normalized, xyz, query_pcl_normalized.transpose(1, 2), features)
    #         feat_list.append(feat)
    #
    #     return torch.cat(feat_list, 1)

    def forward(self, surface_pcl, query_pcl):
        """
        Args:
            surface_pcl (torch.FloatTensor): [B, N, 6]
            query_pcl: (torch.FloatTensor)[B, 3, N]
        Returns:
            feat (torch.FloatTensor): [B, D, N]
        """
        xyz = surface_pcl
        features = surface_pcl.transpose(1, 2)
        feat_list = []
        for sa_module, fp_module in zip(self.SA_modules, self.FP_modules):
            xyz, features = sa_module(xyz, features)

            # from lib.data.mesh_util import save_obj_mesh
            # save_obj_mesh('/media/liaotingting/usb3/mesh.obj', xyz[0].view(-1, 3).cpu().numpy())
            # exit()
            feat = fp_module(query_pcl.transpose(1, 2), xyz, query_pcl, features)
            feat_list.append(feat)

        return torch.cat(feat_list, 1)


if __name__ == '__main__':
    model = PointNet2SSG().cuda()
    pcl = torch.rand(2, 10000, 3).cuda()
    query_points = torch.rand(2, 3, 3000).cuda()
    out = model(pcl, query_points)
    loss = out.mean()