from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F
from architecture.network import Conv2d, ConvTranspose2d, Upsample
import torchvision
import numpy as np

class MCNN_1(nn.Module):
    def __init__(self, bn=False):
        super(MCNN_1, self).__init__()
        self.branch0 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        self.fuse = nn.Sequential(Conv2d( 12, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x = self.branch0(im_data)
        x = self.fuse(x)
        return x

class MCNN_2(nn.Module):
    def __init__(self, bn=False):
        super(MCNN_2, self).__init__()
        self.branch0 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        self.branch1 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        self.fuse = nn.Sequential(Conv2d( 22, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x0 = self.branch0(im_data)
        x1 = self.branch1(im_data)
        x = torch.cat((x0,x1),1)
        x = self.fuse(x)
        return x

class MCNN_3(nn.Module):
    def __init__(self, bn=False):
        super(MCNN_3, self).__init__()
        self.branch0 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        self.branch1 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        self.branch2 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        self.fuse = nn.Sequential(Conv2d(30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x0 = self.branch0(im_data)
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x = torch.cat((x0,x1,x2),1)
        x = self.fuse(x)
        return x

class MCNN_4(nn.Module):
    def __init__(self, bn=False):
        super(MCNN_4, self).__init__()
        self.branch0 = nn.Sequential(Conv2d( 1, 12, 11, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(12, 24, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 12, 9, same_padding=True, bn=bn),
                                     Conv2d(12,  6, 9, same_padding=True, bn=bn))
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        self.fuse = nn.Sequential(Conv2d( 36, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x0 = self.branch0(im_data)
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x0,x1,x2,x3),1)
        x = self.fuse(x)
        return x

class MCNN_4_up(nn.Module):
    def __init__(self, bn=False):
        super(MCNN_4_up, self).__init__()
        self.branch0 = nn.Sequential(Conv2d( 1, 12, 11, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(12, 24, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 12, 9, same_padding=True, bn=bn),
                                     Conv2d(12,  6, 9, same_padding=True, bn=bn))
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        self.fuse = nn.Sequential(Conv2d( 36, 18, 1, same_padding=True, bn=bn))
        self.upscale = nn.Sequential(
            ConvTranspose2d(18, 9, 4, bn=bn)
            , ConvTranspose2d(9, 1, 4, bn=False)
        )

    def forward(self, im_data):
        x0 = self.branch0(im_data)
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x0,x1,x2,x3),1)
        x = self.fuse(x)
        x = self.upscale(x)
        return x


class MCNN_4_skip_conn(nn.Module):
    def __init__(self, bn=False):
        super().__init__()
        self.branch0_0 = Conv2d( 1, 12, 11, same_padding=True, bn=bn)
        self.branch0_1 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(12, 24, 9, same_padding=True, bn=bn)
        )
        self.branch0_2 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(24, 12, 9, same_padding=True, bn=bn)
            , Conv2d(12,  6, 9, same_padding=True, bn=bn)
        )

        self.branch1_0 = Conv2d( 1, 16, 9, same_padding=True, bn=bn)
        self.branch1_1 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(16, 32, 7, same_padding=True, bn=bn)
        )
        self.branch1_2 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(32, 16, 7, same_padding=True, bn=bn)
            , Conv2d(16,  8, 7, same_padding=True, bn=bn)
        )
                                      
        self.branch2_0 = Conv2d( 1, 20, 7, same_padding=True, bn=bn)
        self.branch2_1 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(20, 40, 5, same_padding=True, bn=bn)
        )
        self.branch2_2 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(40, 20, 5, same_padding=True, bn=bn)
            , Conv2d(20, 10, 5, same_padding=True, bn=bn)
        )

        self.branch3_0 = Conv2d( 1, 24, 5, same_padding=True, bn=bn)
        self.branch3_1 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(24, 48, 3, same_padding=True, bn=bn)
        )
        self.branch3_2 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(48, 24, 3, same_padding=True, bn=bn)
            , Conv2d(24, 12, 3, same_padding=True, bn=bn)
        )
        self.reduce_chn_2 = Conv2d(72, 12, 1, same_padding=True, bn=bn)
        self.reduce_chn_1 = Conv2d(144, 24, 1, same_padding=True, bn=bn)
        self.fuse = nn.Sequential(Conv2d( 36, 24, 1, same_padding=True, bn=bn))
        self.up_1 = Upsample(24, 24, 24, 12)
        self.up_2 = Upsample(12, 12, 12, 6)
        self.out = Conv2d(6, 1, 1, same_padding=True, bn=False, relu=True)


    def forward(self, im_data):
        x0_0 = self.branch0_0(im_data)
        x0_1 = self.branch0_1(x0_0)
        x0_2 = self.branch0_2(x0_1)
 
        x1_0 = self.branch1_0(im_data)
        x1_1 = self.branch1_1(x1_0)
        x1_2 = self.branch1_2(x1_1)

        x2_0 = self.branch2_0(im_data)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)

        x3_0 = self.branch3_0(im_data)
        x3_1 = self.branch3_1(x3_0)
        x3_2 = self.branch3_2(x3_1)

        x = torch.cat((x0_2,x1_2,x2_2,x3_2),1)
        x = self.fuse(x)

        x_1 = self.reduce_chn_1(torch.cat((x0_1, x1_1, x2_1, x3_1),1))
        x_2 = self.reduce_chn_2(torch.cat((x0_0, x1_0, x2_0, x3_0),1))

        x = self.up_1(x, x_1)
        x = self.up_2(x, x_2)
        x = self.out(x)

        return x