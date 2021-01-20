from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import architecture.network as network
from architecture.models import MCNN_1, MCNN_2, MCNN_3, MCNN_4
from architecture.GANNet import Discriminator64

class CrowdCounter(nn.Module):
    def __init__(self, model = 'mcnn1', *args, **kwargs):
        super(CrowdCounter, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.args = args
        self.kwargs = kwargs
        self.gan = False
        self.ncritic = 0
        if model == 'mcnn1':
            self.net = MCNN_1()
        elif model == 'mcnn2':
            self.net = MCNN_2()
        elif model == 'mcnn3':
            self.net = MCNN_3()
        elif model == 'mcnn4':
            self.net = MCNN_4()
        elif model == 'mcnn4-gan':
            self.net = MCNN_4()
            self.gan_net = Discriminator64()
            self.gan = True
        else:
            raise RuntimeError("Invalid model: '{}'".format(model))

    def forward(self,  im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        density_map = self.net(im_data)

        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
            self.loss = self.loss_fn(density_map, gt_data)
        return density_map
