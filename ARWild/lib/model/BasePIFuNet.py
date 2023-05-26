import torch.nn as nn

from lib.common.geometry import index, orthogonal, perspective


class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal'):
        """
        :param projection_mode:
        Either orthogonal or perspective.
        It will call the corresponding function for projection.
        nn Loss between the predicted [B, Res, N] and the label [B, Res, N]
        """
        super(BasePIFuNet, self).__init__()
        self.name = 'base'
        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None

        self.preds_normal = None
        self.labels_normal = None

    def forward(self, in_tensor_dict):
        return None

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        return None

    def query(self, xyz, points):
        return None

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.preds[-1]

    def get_error(self, in_tensor_dict):
        '''
        Get the network loss from the last query
        :return: loss term
        '''
        return None
