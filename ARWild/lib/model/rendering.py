import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    ''' Network class containing occupanvy and appearance field

    Args:
        cfg (dict): network configs
    '''

    def __init__(
            self,
            dims,
            skips,
            geometric_init=True,
            radius_init=1.5,
            beta=100):
        super().__init__()

        # position encoding
        # dim = 3
        # dim_in = dim * self.octaves_pe * 2 + dim
        # self.transform_points = PositionalEncoding(L=self.octaves_pe)
        ### geo network
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            if l + 1 in skips:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                # todo add for position encoding
                # elif self.octaves_pe > 0 and l == 0:
                #     torch.nn.init.constant_(lin.bias, 0.0)
                #     torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                #     torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                # elif self.octaves_pe > 0 and l in self.skips:
                #     torch.nn.init.constant_(lin.bias, 0.0)
                #     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                #     torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def infer_occ(self, p):
        pe = self.transform_points(p)
        x = pe
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skips:
                x = torch.cat([x, pe], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x

    def gradient(self, p, feature):
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.infer_occ(p, feature)[..., :1]
            d_output = torch.ones_like(y, device=y.device)
            gradients = torch.autograd.grad(outputs=y,
                                            inputs=p,
                                            grad_outputs=d_output,
                                            create_graph=True,
                                            retain_graph=True, allow_unused=True)[0]
            return gradients.unsqueeze(1)

    def forward(self, p):
        x = self.infer_occ(p)
        return self.sigmoid(x[..., :1])



