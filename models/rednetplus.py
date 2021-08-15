###########################################################################
# Created by: Tashnim Chowdhury
# Email:tchowdh1@umbc.edu   
# Copyright (c) 2021
###########################################################################

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax
import models.resnet as models
from torch.nn.functional import upsample,normalize, interpolate
import torch.nn.functional as F
from .munet import MUNet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, dilated=True, norm_layer=None,
                 base_size=576, crop_size=608, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='./pretrain_models',
                 multi_grid=False, multi_dilation=None):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root,
                                              multi_grid=multi_grid, multi_dilation=multi_dilation)
        elif backbone == 'resnet101':
            self.pretrained = models.resnet101(pretrained=True)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer, root=root,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

        self.layer0 = nn.Sequential(self.pretrained.conv1, self.pretrained.bn1, self.pretrained.relu, self.pretrained.conv2, self.pretrained.bn2, self.pretrained.relu, self.pretrained.conv3, self.pretrained.bn3, self.pretrained.relu, self.pretrained.maxpool)

    def base_forward(self, x):
        x = self.layer0(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union

class ReDNetPlus(BaseNet):
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ReDNetPlus, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = REDNetHead(1024, nclass, norm_layer)
        self.head2 = REDNetHead(2048, nclass, norm_layer)
        
        self.upsample_11 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.conv5c = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                   norm_layer(256),
                                   nn.ReLU())
        
        self.conv5a = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1, bias=False),
                                   norm_layer(128),
                                   nn.ReLU())

        self.sc = Attention_Mod(256)
        self.sa = Attention_Mod(128)

        self.conv52 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                   norm_layer(256),
                                   nn.ReLU())
        self.conv51 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, bias=False),
                                   norm_layer(256),
                                   nn.ReLU())

        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, nclass, 1))
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, nclass, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, nclass, 1))

        self.global_context = GlobalPooling(2048, 256, norm_layer, self._up_kwargs)
        self.munet = MUNet(num_classes=256)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)

        feat1 = self.conv5a(c2)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)
        

        feat2 = self.conv5c(c3)
        
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        sc_conv = self.upsample_11(sc_conv)
        feat_sum = sa_conv+sc_conv
        feat_sum = upsample(feat_sum, imsize, **self._up_kwargs)

        ## mini UNet implementation
        munet_output = self.munet(x)
        munet_output = upsample(munet_output, imsize, **self._up_kwargs)

        final_output_tensor = torch.cat((feat_sum, munet_output), 1)
        final_output = self.conv8(final_output_tensor)

        return final_output


class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h,w), **self._up_kwargs)

class REDNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(REDNetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = Attention_Mod(inter_channels)
        #self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x1):
        feat1 = self.conv5a(x1)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        return sa_output


class Attention_Mod(Module):
    """ Attention Module """
    def __init__(self, in_dim):
        super(Attention_Mod, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            input: 
                x : input feature maps (B x C x H x W)
            returns:
                out : attention value + input feature
                attention: B x (HxW) x (HxW)
        """

        m_batchsize, m_channel, m_height, m_width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, m_width*m_height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, m_width*m_height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, m_width*m_height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, m_channel, m_height, m_width)

        out = self.gamma*out + x
        return out





