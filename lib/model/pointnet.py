import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pytorch3d.ops.knn import knn_points, knn_gather

pointnet_kwarg = {
    'only_global_feat': False,
    'local_encoder_dims': [3, 64],
    'global_encoder_dims': [64, 128, 64],
    'feature_transform': True,
    'feat_transform_kwarg': {
        'k': 64,
        'encoder_channels': [64, 128, 256],
        'decoder_channels': [256, 512, 256],
    }
}


def ConvBlock(in_ch, out_ch, use_relu=True):
    net = [nn.Conv1d(in_ch, out_ch, 1)]
    if use_relu:
        # net.append(nn.BatchNorm1d(out_ch))
        net.append(nn.ReLU(True))
    return nn.Sequential(*net)


def FCLayers(dims, out_linear):
    net = []
    for i in range(len(dims) - 1):
        use_relu = i < len(dims) - 2 or not out_linear
        net.append(ConvBlock(dims[i], dims[i + 1], use_relu))
    return nn.Sequential(*net)


class STNkd(nn.Module):
    def __init__(self, k, encoder_channels, decoder_channels):
        super(STNkd, self).__init__()
        self.k = k
        self.encoder = FCLayers([k] + encoder_channels, out_linear=False)
        self.decoder = FCLayers(decoder_channels + [k * k], out_linear=True)

    def forward(self, input):
        """
        Args:
            input: [B, dim, N]
        Returns:
            out: [B, dim, N]
            trans: [B, K, K]
        """
        trans = self.encoder(input)
        trans = torch.max(trans, 2, keepdim=True)[0]
        trans = self.decoder(trans)

        iden = Variable(torch.eye(self.k).unsqueeze(0).float())
        trans = trans.view(-1, self.k, self.k) + iden.to(trans.device)
        out = torch.bmm(input.transpose(2, 1), trans).transpose(2, 1)
        return out, trans


class PointNetEncoder(nn.Module):
    def __init__(self, local_encoder_dims, global_encoder_dims,
                 only_global_feat=False, feature_transform=False, feat_transform_kwarg=None):
        super(PointNetEncoder, self).__init__()
        self.only_global_feat = only_global_feat
        self.local_encoder = FCLayers(local_encoder_dims, out_linear=False)
        self.global_encoder = FCLayers(global_encoder_dims, out_linear=True)
        self.feat_transform_net = STNkd(**feat_transform_kwarg) if feature_transform else None

    def forward(self, point_cloud, query_points, K=3):
        """
        Args:
            point_cloud: [B, 3, N]
            query_points: [B, 3, N]
        Returns:
        """
        N = point_cloud.size(2)
        feat = self.local_encoder(point_cloud)

        if self.feat_transform_net is not None:
            feat, _ = self.feat_transform_net(feat)
            print(feat.shape)
            exit()
        local_feat = feat
        feat = self.global_encoder(feat)
        feat = torch.max(feat, 2)[0].unsqueeze(2)  # [B, dim, 1]

        print(feat.shape)

        if not self.only_global_feat:
            feat = torch.cat([feat.repeat(1, 1, N), local_feat], 1)
        print(feat.shape)
        exit()
        # find nearest neighbors
        dist, ids, _ = knn_points(query_points.transpose(1, 2).contiguous(),
                                  point_cloud.transpose(1, 2).contiguous(), K=K)
        feat = knn_gather(feat.transpose(1, 2), ids)  # (B, N, K, Dim)

        # feat = torch.cat([dist.unsqueeze(-1), feat], -1)
        return feat.mean(2).transpose(1, 2).contiguous()



if __name__ == '__main__':
    net = PointNetEncoder(**pointnet_kwarg)
    points = torch.rand(1, 3, 6890)
    query_points = torch.rand(1, 3, 1000)
    out = net(points, query_points)
    print(out.shape)
    exit()
