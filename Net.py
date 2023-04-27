

######################################################################################
#U-Net: Convolutional Networks for BiomedicalImage Segmentation
#Paper-Link: https://arxiv.org/pdf/1505.04597.pdf
######################################################################################
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

import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from module.efficientnet_pytorch import EfficientNet
from module.attention import CAM_Module ,fuse,PAM_Module,fuse_position
import segmentation_models_pytorch as smp


__all__ = ["UNet"]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.model = smp.Unet(   encoder_name="efficientnet-b7" , 
            encoder_depth = 4,# choose encoder, e.g. mobilenet_v2 or efficientnet-b
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=5,                      # model output channels (number of classes in your dataset)
            decoder_channels = (256,128,64,32) 
        )
        
        self.onepix_attention = return_pix()

    def forward(self, inputs,targets,output_teature,mode,output_u):
        output = self.model(inputs)
        

        #############　推論時のRA-Module　################
        if mode == 'test' or mode =='val':
            index = check_pix_test(output,output_u) #確信度を使った時の失敗画素取得
            index_len = len(index)
            make_attention = output
            self_teature,attention_s,index_len_max = self.onepix_attention(index, index_len, make_attention)
            return output,self_teature, index,index_len

        #############　学習のRA-Module　################
        index = check_pix(output,targets)
        index_len = len(index)
        index.extend(index)
        index=set(index)
        make_attention = output
        self_teature,attention_s,index_len_max = self.onepix_attention(index, index_len, make_attention)

        return output,self_teature,index_len



def check_pix(output_student,target):
    m_batchsize, C, height, width = output_student.size()
    #ターゲットをnumpyに
    target = target.detach().cpu().numpy()
    target = np.array(target)

    #ラベルとの比較で成功か失敗かを取得
    fix_pixel_base = output_student.detach().cpu().numpy()
    fix_pixel_base = np.array(fix_pixel_base)
    fix_pixel_base = np.argmax(fix_pixel_base,axis=1)
    fix_pixel = np.zeros((m_batchsize, 256, 256)).astype(np.float32)
    fix_pixel = fix_pixel_base - target #   正解は０　失敗が0以外になる

    index = list(zip(*np.where((fix_pixel != 0) ))) #確信度が0.7いか　かつ　失敗している画素のインデクス（確信をもって失敗している奴）
    print(len(index))

    return index

def check_pix_test(output_student,output_teacher):
    m_batchsize, C, height, width = output_student.size()

    #teacherとの比較で成功か失敗かを取得
    fix_pixel_base =output_student.detach().cpu().numpy()
    fix_pixel_base = np.array(fix_pixel_base)
    fix_pixel_base = np.argmax(fix_pixel_base,axis=1)
    fix_pixel_t =output_teacher.detach().cpu().numpy()
    fix_pixel_t = np.array(fix_pixel_t)
    fix_pixel_t = np.argmax(fix_pixel_t,axis=1)
    fix_pixel = np.zeros((m_batchsize, 256, 256)).astype(np.float32)
    fix_pixel = fix_pixel_base - fix_pixel_t  #   正解は０　失敗が0以外になる


    #ピクセルが0以外のところのインデクスを取得
    index = list(zip(*np.where(fix_pixel  != 0) )) #確信度が0.5以下　
    print(len(index))

    return index



####################確信度を利用した場合の失敗画素特定方法########################

def check_pix_kakusindo(output_student,target):
    m_batchsize, C, height, width = output_student.size()
    #ターゲットをnumpyに
    target = target.detach().cpu().numpy()
    target = np.array(target)

    #クラスごとの確信度を出力
    output_student = output_student.detach().view(m_batchsize,5,width*height)
    Softmax = nn.Softmax(dim=1)
    output_student_probability = Softmax(output_student)
    output_student_probability =output_student_probability.detach().cpu().numpy()
    output_student_probability = np.array(output_student_probability)

    #確率最大値のラベルを出力
    output_student_max = np.argmax(output_student_probability, axis=1)
    #print(output_student_max)
    pixel_student = np.zeros((m_batchsize, 256, 256)).astype(np.float32)
    
    #確信度が0.5異常の場所の値を1にする
    for batch in range(m_batchsize):
        for i in range(256 * 256):
            F_map = output_student_probability[batch][output_student_max[batch][i]][i]
            
            if 0.7>F_map :
                pixel_student[batch][i // 256][i % 256] = 1

    #ピクセルが0以外のところのインデクスを取得
    index = list(zip(*np.where(pixel_student  == 1) )) #確信度が0.5以下　
    print(len(index))

    return index


class return_pix(Module):
    def __init__(self):
        super(return_pix, self).__init__()
        in_dim = 5
        self.gamma = Parameter(torch.zeros(1))
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.conv_last = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1).to(0)
        self.softmax = Softmax(dim=-1)
    
    def forward(self,index,index_len,x):
            gpu = 0
            m_batchsize, C, height, width = x.size()
            at_predict = []
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
            x_val = self.value_conv(x)
            print('{:.9f}'.format(self.gamma.item()))

            ##################失敗画素の上限を決める場所#########################
            index_len2 = 25000
            if index_len2 <= index_len:
                index_len = 25000
            index_random = random.sample(index, index_len)


            for i in tqdm(reversed(range(index_len)),total=index_len,desc="adress late"):
                
                num = index_random[i]
                for j in range(5):
                    select_pix_A[i][0][j][0] = x_query[num[0]][j][num[1]][num[2]]#例[0,0,11]のアドレスのデータを取得 size 256

            for i in tqdm(reversed(range(index_len)),total=index_len,desc="attention late"):
                num = index_random[i]
                select_pix = select_pix_A[i]
                proj_query  = select_pix.permute(0,2,1).to(0)        
                proj_key    = x_key[num[0]].view(1,5, width*height).to(0)               
                energy_S      = torch.bmm(proj_query,proj_key)
                attention_S = self.softmax(energy_S)
                proj_value = x_val[num[0]].view(1, 5, width*height)
                outS = torch.bmm(proj_value, attention_S.permute(0, 2, 1))
                attention_s = energy_S.view(256,256).detach().cpu().numpy()
                at_predict.append(attention_s)
                for j in range(5):
                    y[num[0]][j][num[1]][num[2]] = (self.gamma+0.1) * outS[0][j][0] + x[num[0]][j][num[1]][num[2]]
                    
            return y,at_predict,index_len


