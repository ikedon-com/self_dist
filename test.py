#coding: utf-8
##### ライブラリ読み込み #####
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
from Net import UNet
import matplotlib.pyplot as plt
import seaborn as sns
from utile.classweight import class_weight,class_weight_4
from utile.dataloader import FlyCellDataLoader_crossval
from utile.IoU import IoU
from utile.save import Save_image
from utile.train_val import train,val,test
import segmentation_models_pytorch as smp

if __name__ == '__main__':
    ####################### コマンド設定 #######################
    parser = argparse.ArgumentParser(description='Segmentation')
    # ミニバッチサイズ(学習)指定
    parser.add_argument('--batchsize', '-b', type=int, default=2,
                        help='Number of images in each mini-batch')
    # ミニバッチサイズ(テスト)指定
    parser.add_argument('--Tbatchsize', '-t', type=int, default=1,
                        help='Number of images in each mini-batch')
    # 学習回数(epoch)指定
    parser.add_argument('--num_epochs', '-e', type=int, default=4000,
                        help='Number of sweeps over the dataset to train')
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result_class5',
                        help='Directory to output the result')
    # GPU指定
    parser.add_argument('--gpu', '-g', type=str, default=0,
                        help='Directory to output the result')
    parser.add_argument('--gpu4', '-g_4', type=str, default=1,
                        help='Directory to output the result')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')

    parser.add_argument("--rootdir", type=str, default='Dataset/')
    #parser.add_argument("--batchsize", "-b", default=16, type=int)
    parser.add_argument("--iter", default=12, type=int)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--val_area", default=2, type=int, help='cross-val test area [default: 5]')
    parser.add_argument('--debug', action='store_true', help='use cuda?')
    args = parser.parse_args()

    ##################### GPU設定 #######################
    gpu_flag = args.gpu
    gpu_flag2 = args.gpu4
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    cudnn.benchmark = True
    device = torch.device("cuda:{}".format(gpu_flag))  
    device2 = torch.device("cuda:{}".format(gpu_flag))  
    device3 = torch.device("cuda:{}".format(gpu_flag))  

    ################# 出力フォルダ作成 ####################
    if not os.path.exists("{}".format(args.out)):
      	    os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "model")):
      	    os.mkdir(os.path.join("{}".format(args.out), "model"))
    if not os.path.exists(os.path.join("{}".format(args.out), "image")):
      	    os.mkdir(os.path.join("{}".format(args.out), "image"))
    PATH_1 = "{}/trainloss.txt".format(args.out)
    PATH_2 = "{}/testloss.txt".format(args.out)
    PATH_3 = "{}/IoU.txt".format(args.out)
    PATH_4 = "{}/IoU_test.txt".format(args.out)
    PATH_5 = "{}/IoU_test4.txt".format(args.out)
    PATH_mse = "{}/mse_loss.txt".format(args.out)

    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass
    with open(PATH_4, mode = 'w') as f:
        pass
    with open(PATH_5, mode = 'w') as f:
        pass
    with open(PATH_mse, mode = 'w') as f:
        pass

    ################ データ読み込み+初期設定 ###############
    # 初期値
    sample = 0
    sample4 = 0
    tsample = 0
    tsample4 = 0
    tmm4 = 0
    n_class = 5
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    # データローダー作成
    ds_train = FlyCellDataLoader_crossval(rootdir=args.rootdir, val_area=args.val_area, split='train', iteration_number=args.batchsize*args.iter)
    ds_val = FlyCellDataLoader_crossval(rootdir=args.rootdir, val_area=args.val_area, split='val')
    ds_test = FlyCellDataLoader_crossval(rootdir=args.rootdir, val_area=args.val_area, split='test')
    
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batchsize, shuffle=True, num_workers=args.threads)
    #train_loader_w = torch.utils.data.DataLoader(ds_W, batch_size=1, shuffle=False, num_workers=args.threads)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=8 , shuffle=False, num_workers=args.threads)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=args.threads)



    model = UNet()

    model_u = smp.Unet(   encoder_name="efficientnet-b7" , 
        encoder_depth = 4,# choose encoder, e.g. mobilenet_v2 or efficientnet-b
        encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=5,                      # model output channels (number of classes in your dataset)
        decoder_channels = (256,128,64,32) 
    )




    model_dict = torch.load('result_val2_kakusindo_orhuseikaigaso/model/model_bestiou_test.pth',map_location=device)
    model.load_state_dict(model_dict)
    model_dict = torch.load('load_model/model_bestiou_ALL.pth',map_location=device)
    model_u.load_state_dict(model_dict)    

    if args.cuda:
        model = model.to(device) #モデルをGPUモードに
        model_u = model_u.to(device) #モデルをGPUモードに
        model_maku = model_maku.to(device) #モデルをGPUモードに
        model_mitokon = model_mitokon.to(device2) #モデルをGPUモードに
        model_syna = model_syna.to(device2) #モデルをGPUモードに
        model_naimaku = model_naimaku.to(device2) #モデルをGPUモードに
        model_back = model_back.to(device2) #モデルをGPUモードに




    epoch = 0
    gamma = 0

    tmm ,tm1,tm2,tm3,tm4,tm5= test(epoch,test_loader,model,args,device,n_class,gamma)
    

    print(tmm)
    print(tm1)
    print(tm2)
    print(tm3)
    print(tm4)
    print(tm5)
    print(attmm)
    print(attm1)
    print(attm2)
    print(attm3)
    print(attm4)
    print(attm5)

    



