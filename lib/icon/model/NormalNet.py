import torch
import torch.nn as nn
import functools


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
            activation
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', last_op=nn.Tanh()):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if last_op is not None:
            model += [last_op]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance', last_op=nn.Tanh()):
    norm_layer = get_norm_layer(norm_type=norm)

    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer,
                           last_op=last_op)

    netG.apply(weights_init)
    return netG


class NormalNet(nn.Module):
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

    def __init__(self, in_dim):
        super(NormalNet, self).__init__()
        self.netF = define_G(in_dim, 3, 64, 4)
        self.netB = define_G(in_dim, 3, 64, 4)

    def forward(self, image, mask, T_normal_F=None, T_normal_B=None):
        """
        Args:
            T_normal_B:
            image: [B, 3, 512, 512],
            mask: [B, 1, 512, 512]
            T_normal_F: [B, 3, 512, 512],
            T_normal_B: [B, 3, 512, 512],
        Returns:
        """
        if T_normal_F is not None and T_normal_B is not None:
            nmlF = self.netF(torch.cat([image, T_normal_F], 1))
            nmlB = self.netB(torch.cat([image, T_normal_B], 1))
        else:
            nmlF = self.netF(image)
            nmlB = self.netB(image)

        nmlF = nmlF * mask
        nmlB = nmlB * mask

        return nmlF, nmlB


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


def get_normal_model(type='icon'):
    if type == 'icon':
        in_dim = 6
        ckpt_path = './data/icon_data/normal.ckpt'
    elif type == 'pifuhd':
        in_dim = 3
        ckpt_path = '/media/liaotingting/usb2/projects/PIFuSOTA/checkpoints/pifuhd/pifuhd.pt'
    else:
        raise TypeError('normal network type must be icon or pifuhd')

    model = NormalNet(in_dim=in_dim)

    normal_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    normal_dict = normal_dict['state_dict'] if 'state_dict' in normal_dict else normal_dict['model_state_dict']
    for key in normal_dict.keys():
        normal_dict = rename(normal_dict, key, key.replace("netG.", ""))

    model_dict = model.state_dict()
    normal_dict = {
        k: v
        for k, v in normal_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(normal_dict)
    model.load_state_dict(model_dict)
    print(f"Resume normal model from {ckpt_path}")
    model.training = False
    model.eval()

    del normal_dict
    del model_dict
    return model

