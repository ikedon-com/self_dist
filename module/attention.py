###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        """

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        """

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(1)
        self.gamma = Parameter(torch.zeros(0))

        self.softmax = Softmax(dim=-1)
        self.x_changeconv = Conv2d(in_channels=224, out_channels=5, kernel_size=1).to(1)   
        self.fuse = fuse_position(5)

    def forward(self, x):
        x = x.cuda(0)

      #  print(x.shape)
        m_batchsize, C, height, width = x.size()

        if C == 224:
            x = self.x_changeconv(x)
            x = F.interpolate(x,(32,32),mode='bilinear')
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1).to(1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height).to(1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        self.fuse(x,attention)

        return attention

class fuse_position(Module):
    def __init__(self, in_dim):
        super(fuse_position, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1)).to(1)
        self.softmax  = Softmax(dim=-1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(1)        

    def forward(self,x,attention):
        x = x.cuda(0)
        attention =attention.cuda(0)
        m_batchsize, C, height, width = x.size()

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x

        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        self.sigmoid = Sigmoid()
    def forward(self,x):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)

        energy_new = self.sigmoid(energy_new)

        """
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        """
        return attention,energy_new

class fuse(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(fuse, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1)).to(1)
        self.softmax  = Softmax(dim=-1)
        
    def forward(self,x,attention):
        x = x.cuda(0)
        attention =attention.cuda(0)
        m_batchsize, C, height, width = x.size()
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        

        out = self.gamma*out + x

        return out