from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F
from architecture.network import Conv2d, ConvTranspose2d
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
        self.fuse = nn.Sequential(Conv2d( 36, 18, 1, same_padding=True, bn=bn))
        self.upsample = nn.Sequential(
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
        x = self.upsample(x)
        return x