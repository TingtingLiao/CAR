import torch
import torch.nn as nn
from .net_util import last_hyper_init, MetaMLP, weights_normal_init
from collections import OrderedDict
from torchmeta.modules import MetaModule, MetaSequential


class HyperNet(MetaModule):
    '''Builds a hypernetwork that predicts a fully connected neural network.
    '''
    def __init__(self, encoder, sdf_net, num_hidden_layers, hidden_ch, skip_in, hyper_in_chs,
                 init_type='normal'):
        super(HyperNet, self).__init__()
        self.encoder = encoder
        self.nets = []
        self.names = []
        self.param_shapes = []

        # hyper_in_chs = {name: param.size(0) for name, param in encoder_named_parameters if 'weight' in name}
        i = 0
        for name, param in sdf_net.meta_named_parameters():
            if 'weight' in name:
                shape = (param.size(0), param.size(1) + 1)
                out_ch = int(torch.prod(torch.tensor(shape)))
                in_ch = hyper_in_chs[i]
                i += 1
                dims = [in_ch] + [hidden_ch] * num_hidden_layers + [out_ch]
                net = MetaMLP(dims=dims, geometric_init=False, beta=-1, skip_in=skip_in)

                net.apply(weights_normal_init)
                if init_type == 'normal':
                    with torch.no_grad():
                        net.net[-1].weight *= 1e-1
                elif init_type == 'last_zero':
                    net.net[-1].apply(last_hyper_init)
                    # net.apply(last_hyper_init)

                self.nets.append(net)
                self.names.append(name)
                self.param_shapes.append(shape)

        self.nets = MetaSequential(*self.nets)

    def forward(self, input, hyper_params=None, return_feats=False, feat_list=None):
        '''
        Args:
            input: Input to hypernetwork.
            hyper_params: parameters
            return_feats: bool return features from encoder if true
        Return

        '''
        # feats = self.encoder(input, params=self.get_subdict(hyper_params, 'encoder'), return_feats=True)
        feats = self.encoder(input, return_feats=True)
        sdf_params = OrderedDict()
        for i, (name, net, param_shape, feat) in enumerate(zip(self.names, self.nets, self.param_shapes, feats)):
            bias_name = name.replace('weight', 'bias')
            if feat.dim() == 3:
                feat = feat.mean(1)
            elif feat.dim() == 4:
                feat = feat.mean(2).mean(2)
            else:
                print(feat.shape)
                exit()
                raise ValueError
            # params = net(
            #     feat, params=self.get_subdict(self.get_subdict(hyper_params, 'nets'), str(i))
            # ).reshape((-1,) + param_shape)

            if feat_list is not None:
                feat = torch.cat([feat, feat_list[i].mean(1)], -1)

            params = net(feat).reshape((-1,) + param_shape)
            sdf_params[name] = params[..., 1:]
            sdf_params[bias_name] = params[..., 0]

        if return_feats:
            return sdf_params, feats

        return sdf_params
