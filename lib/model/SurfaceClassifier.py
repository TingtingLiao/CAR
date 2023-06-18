import torch
import torch.nn as nn
import numpy as np
from vit_pytorch.vit import Transformer, repeat
import torch.nn.functional as F


class ImplicitNet(nn.Module):
    def __init__(
            self,
            dims,
            num_views,
            skip_in=(),
            geometric_init=True,
            radius_init=1.5,
            beta=100,
            octaves_pe=0,
            last_op=None
    ):
        # geometric initialization centered not at the origin
        super(ImplicitNet, self).__init__()

        if octaves_pe > 0:
            dims[0] += 3 * octaves_pe * 2 - 3
            self.position_encoding = PositionalEncoding(octaves_pe)

        d_in = dims[0]

        self.num_views = num_views
        self.beta = beta
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.octaves_pe = octaves_pe

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                elif octaves_pe > 0 and layer == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif octaves_pe > 0 and layer in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -d_in:], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU(inplace=True)
        self.last_op = last_op

    def forward(self, input):
        """
        Args:
            input: [B, D, N]

        Returns:

        """
        if self.octaves_pe > 0:
            feat, pts = torch.split(input, [input.shape[1] - 3, 3], dim=1)
            pts = self.position_encoding(pts.transpose(1, 2)).transpose(1, 2)
            x = torch.cat([feat, pts], dim=1).transpose(1, 2)
        else:
            x = input.transpose(1, 2)

        tmp = x

        for layer in range(0, self.num_layers - 1):

            if layer in self.skip_in:
                x = torch.cat([x, tmp], -1) / np.sqrt(2)

            lin = getattr(self, "lin" + str(layer))

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

            if self.num_views > 1 and layer == self.num_layers // 2:
                x = x.view(-1, self.num_views, x.shape[1], x.shape[2]).mean(dim=1)
                tmp = tmp.view(-1, self.num_views, tmp.shape[1], tmp.shape[2]).mean(dim=1)

        if self.last_op is not None:
            x = self.last_op(x)

        return x.transpose(1, 2)


class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.last_op = last_op
        filter_channels = filter_channels

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](y if i == 0 else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(-1, self.num_views, y.shape[1], y.shape[2]).mean(dim=1)
                tmpy = feature.view(-1, self.num_views, feature.shape[1], feature.shape[2]).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)
        return y


class ViT(nn.Module):
    def __init__(self, *, num_views, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0.):
        # super().__init__()
        super(ViT, self).__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.num_views = num_views
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        _, C, N = img.shape
        x = img.contiguous().view(-1, self.num_views, C, N)
        x = x.permute(0, 3, 1, 2).contiguous().view(-1, self.num_views, C)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        x = x.view(-1, N, x.shape[1]).permute(0, 2, 1).contiguous()
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, n_harmonic_functions=10, omega0=1):
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).contiguous().view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)
