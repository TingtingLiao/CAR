import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dataroot', type=str, default='./data',
                            help='path to images (data folder)')

        g_data.add_argument('--loadSize', type=int, default=512, help='load size of input image')
        g_data.add_argument('--input_im', '-ii', type=str, default='rgb', help='input image type [rgb, normal]')
        g_data.add_argument('--max_iter', type=int, default=1e5, help='input image type [rgb, normal]')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='',
                           help='name of the experiment. It decides where to store samples and models')
        g_exp.add_argument('--debug', type=int, default=1, help='debug mode or not')
        parser.add_argument('--projection_mode', default='orthogonal', help='reload model')
        parser.add_argument('--reload', action='store_true', help='reload model')
        g_exp.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')
        g_exp.add_argument('--random_multiview', action='store_true', help='Select random multiview combination.')
        g_exp.add_argument('--eval', action='store_true', help='Select random multiview combination.')
        g_exp.add_argument('--add_noise', action='store_true', help='Add noise .')
        g_exp.add_argument('--noise_std', default=0.01, type=float, help='The standard of added noise.')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', nargs='+', default=[0, 1, 2, 3], type=int, help='gpu ids')

        g_train.add_argument('--num_threads', default=4, type=int, help='# sthreads for loading data')
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        # g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        g_train.add_argument('--pin_memory', default=True, type=bool, help='pin_memory')

        g_train.add_argument('--batch_size', type=int, default=4, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--learning_rateC', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=6000, help='num epoch to train')
        # g_train.add_argument('--in_loop_epoch', type=int, default=10, help='num epoch to train')

        g_train.add_argument('--freq_plot', type=int, default=1, help='freqency of the error plot')
        g_train.add_argument('--freq_save', type=int, default=1, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_save_ply', type=int, default=1, help='freqency of the save ply')

        g_train.add_argument('--no_gen_mesh', action='store_true')
        g_train.add_argument('--no_num_eval', action='store_true')

        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')
        g_test.add_argument('--test_folder_path', type=str, default=None, help='the folder of test image')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma', type=float, default=0.05, help='perturbation standard deviation for positions')
        g_sample.add_argument('--sigma_lr', type=float, default=0.4,
                              help='perturbation standard deviation for positions')
        parser.add_argument('--sigma_schedule', type=int, nargs='+', default=[1000, 1500],
                            help='Decrease sigma at these epochs.')

        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points')
        g_sample.add_argument('--num_sample_skin', type=int, default=0, help='# of sampling points')

        g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        g_model.add_argument('--norm_color', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')

        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[259, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_surface', nargs='+', default=[513, 1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of color mlp')
        g_model.add_argument('--mlp_dim_skin', nargs='+', default=[515, 1024, 512, 256, 128, 25],
                             type=int, help='# of dimensions of color mlp')

        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')

        # for train
        parser.add_argument('--random_flip', default=True, type=bool, help='if random flip')
        parser.add_argument('--random_trans', default=True, type=bool, help='if random flip')
        parser.add_argument('--random_scale', default=True, type=bool, help='if random flip')
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--schedule', type=int, nargs='+', default=[800],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--color_loss_type', type=str, default='l1', help='mse | l1')

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
        parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh')
        parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
        parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
        parser.add_argument('--num_gen_mesh_test', type=int, default=1,
                            help='how many meshes to generate during testing')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--load_netS_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='/media/liaotingting/usb2/tmp', help='path to save results ply')
        parser.add_argument('--load_checkpoint_path', type=str, help='path to save results ply')
        parser.add_argument('--single', type=str, default='', help='single data for training')
        # for single image reconstruction
        parser.add_argument('--mask_path', type=str, help='path for input mask')
        parser.add_argument('--img_path', type=str, help='path for input image')

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
        group_aug.add_argument('--aug_bri', type=float, default=0.4, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.4, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.4, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.0, help='augmentation hue')
        group_aug.add_argument('--aug_blur', type=float, default=1.0, help='augmentation blur')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt


def get_skin_net_default_option():
    opt = BaseOptions().parse()
    opt.num_sample_inout = 0
    opt.num_sample_skin = 5000
    opt.num_epoch = 2000
    opt.last_op = 'softmax'
    opt.num_views = 4
    opt.learning_rate = 1e-4

    opt.gpu_ids = [0, 1, 2, 3]
    opt.gpu_id = 1
    opt.batch_size = 4 // opt.num_views
    opt.mlp_dim = [280, 1024, 512, 256, 128, 25]
    opt.name = 'SWNet-PE-4'
    opt.embedding_type = 'position_encoding'
    opt.netS_checkpoint_path = './checkpoints/SWNet-PE-%d/netS' % opt.num_views
    opt.netG_checkpoint_path = None
    opt.average_fusion = True
    return opt


def get_geo_net_default_option():
    opt = BaseOptions().parse()
    opt.num_sample_skin = 0
    opt.num_sample_inout = 5000
    opt.num_epoch = 3600
    opt.last_op = 'sigmoid'
    opt.gpu_ids = [0, 1, 2, 3]
    opt.batch_size = 4 // opt.num_views
    opt.mlp_dim = [280, 1024, 512, 256, 128, 1]
    opt.name = 'SRNet-PE-4'
    opt.embedding_type = 'position_encoding'
    opt.netS_checkpoint_path = './checkpoints/%s/netS' % opt.name
    opt.netG_checkpoint_path = None
    opt.average_fusion = True
    opt.sigma = 0.05
    return opt


def get_reconstruction_net_default_option():
    opt = BaseOptions().parse()
    opt.name = 'SRNet-HE-Fusion'
    opt.num_sample_skin = 0
    opt.num_sample_inout = 10000
    opt.num_epoch = 1000
    opt.learning_rate = 1e-4
    opt.mlp_dim = [280, 1024, 512, 256, 128, 25]
    opt.mlp_dim_surface = [561, 1024, 512, 256, 128, 1]
    opt.embedding_type = 'position_encoding'
    # opt.name = 'SRNet-HE-Fusion'
    opt.netS_checkpoint_path = './checkpoints/SWNet-PE-4/netS'
    opt.netG_checkpoint_path = './checkpoints/SRNet-HE-Fusion/netG'
    opt.last_op = 'softmax'
    opt.average_fusion = False
    opt.gpu_id = 2
    opt.sigma = 0.05
    opt.batch_size = 1
    opt.schedule = [500]
    return opt


def get_reconstruction_wofeat_default_option():
    opt = BaseOptions().parse()
    opt.name = 'SRNet-HE-Fusion'
    opt.num_sample_skin = 0
    opt.num_sample_inout = 10000
    opt.num_epoch = 1000
    opt.learning_rate = 1e-4
    opt.mlp_dim = [280, 1024, 512, 256, 128, 25]
    opt.mlp_dim_surface = [561, 1024, 512, 256, 128, 1]
    opt.embedding_type = 'depth'
    opt.name = 'SRNet-depth'
    opt.netS_checkpoint_path = './checkpoints/SWNet-PE-4/netS'
    opt.netG_checkpoint_path = './checkpoints/SRNet-HE-Fusion/netG'
    opt.last_op = 'softmax'
    opt.average_fusion = False
    opt.gpu_id = 2
    opt.sigma = 0.05
    opt.batch_size = 1
    opt.schedule = [500]
    return opt
