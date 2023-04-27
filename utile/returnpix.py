import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import time
import cv2
import os
import argparse
import random
from sklearn.metrics import confusion_matrix
from module.EFF import Efficient
from module.hiramatsUnet import HiramatsuUNET

import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm


def check_pix(output_student,output_teature,target):
    m_batchsize, C, height, width = output_student.size()

    #output_student = output_student.view(m_batchsize, -1, width*height)
    #output_teature = output_teature.view(m_batchsize, -1, width*height)
    #target = target.view(m_batchsize,  width*height)

    output_student =output_student.detach().cpu().numpy()
    output_teature = output_teature.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    output_student = np.array(output_student)
    output_teature = np.array(output_teature)
    target = np.array(target)
    #print(target.shape)

    # 確率最大値のラベルを出力
    output_student = np.argmax(output_student,axis=1)
    output_teature = np.argmax(output_teature,axis=1)

    pixel = np.zeros((m_batchsize,256,256)).astype(np.float32)
    pixel_teature = np.zeros((m_batchsize,256,256)).astype(np.float32)
    pixel_student = np.zeros((m_batchsize,256,256)).astype(np.float32)


    pixel_teature = output_teature - target
    pixel_student = output_student - target

    
    #ピクセルが0以外のところのインデクスを取得
    index_teature = list(zip(*np.where(pixel_teature != 0)))
    index_student = list(zip(*np.where(pixel_student != 0)))

    index =list(zip(*np.where((pixel_student != 0) & (pixel_teature == 0))))
    #pprint.pprint(len(index))

    return index

class return_pix(Module):
    
    def __init__(self,args):
        super(return_pix, self).__init__()
        in_dim = 5
        gpu = args.gpu
        self.is_debug = args.debug
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.query_conv_T = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.key_conv_T = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.conv_last = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.gamma = Parameter(torch.zeros(1)).to(0)

        self.softmax = Softmax(dim=-1)
    
    def forward(self,args,index,index_len,x,x_teature,gamma):
            gpu = 0
            m_batchsize, C, height, width = x.size()

            select_pix_A = np.zeros((index_len,1,5,1)).astype(np.float32)
            select_pix_A = torch.from_numpy(select_pix_A).to(0)
            select_pix_TA = np.zeros((index_len,1,5,1)).astype(np.float32)
            select_pix_TA = torch.from_numpy(select_pix_TA).to(0)
            attention_s = np.zeros((index_len,1,5,1)).astype(np.float32)
            attention_s = torch.from_numpy(attention_s).cuda(0)
            attention_t = np.zeros((index_len,1,5,1)).astype(np.float32)
            attention_t = torch.from_numpy(attention_t).cuda(0)
            y =x.clone().detach()            
            w = torch.zeros(index_len)
            losssum=0    
            x_query = self.query_conv(x)
            x_key = self.key_conv(x)
            x_query_T = self.query_conv(x_teature)
            x_key_T = self.key_conv(x_teature)
            x_val = self.value_conv(x)


            index_len = 10000
            index_random = random.sample(index,index_len)

            for i in tqdm(reversed(range(index_len)),total=index_len,desc="adress late"):
                
                num = index_random[i]
                #print(i)
                for j in range(5):
                    select_pix_A[i][0][j][0] = x_query[num[0]][j][num[1]][num[2]]#例[0,0,11]のアドレスのデータを取得 size 256
                    #select_pix_TA[i][0][j][0] = x_query_T[num[0]][j][num[1]][num[2]]#例[0,0,11]のアドレスのデータを取得 size 256


            for i in tqdm(reversed(range(index_len)),total=index_len,desc="attention late"):
                num = index_random[i]
                select_pix = select_pix_A[i]
                #select_pix_T = select_pix_TA[i]
                proj_query  = select_pix.permute(0,2,1).to(0)          
                proj_key    = x_key[num[0]].view(1,5, width*height).to(0)                 
                energy_S      = torch.bmm(proj_query,proj_key)
                attention_S = self.softmax(energy_S)
                proj_value = x_val[num[0]].view(1, 5, width*height)
                outS = torch.bmm(proj_value, attention_S.permute(0, 2, 1))
                #print(outS)

                for j in range(5):
                    y[num[0]][j][num[1]][num[2]] = gamma * outS[0][j][0] + x[num[0]][j][num[1]][num[2]]
                    
            print(gamma)
                    




            return y,y
