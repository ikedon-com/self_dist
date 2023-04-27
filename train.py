#coding: utf-8
##### ライブラリ読み込み #####
import numpy as np
import torch

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import argparse
import random
from Net import UNet
from utile.classweight import class_weight
from utile.dataloader import FlyCellDataLoader_crossval
from utile.train_val import train,val,test
import segmentation_models_pytorch as smp
from thop import profile

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
    parser.add_argument('--num_epochs', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='Result_Default',
                        help='Directory to output the result')
    # GPU指定
    parser.add_argument('--gpu', '-g', type=str, default=0,
                        help='Directory to output the result')
    parser.add_argument('--gpu4', '-g_4', type=str, default=1,
                        help='Directory to output the result')
    parser.add_argument('--cuda',default='cuda', action='store_true', help='use cuda?')
    parser.add_argument('--debug', action='store_true', help='use cuda?')
    parser.add_argument("--rootdir", type=str, default='Dataset/')

    parser.add_argument("--iter", default=24, type=int)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--val_area", default=2, type=int, help='cross-val test area [default: 5]')

    args = parser.parse_args()

    ##################### GPU設定 #######################
    gpu_flag = args.gpu
    device = torch.device("cuda:{}".format(gpu_flag))  
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    cudnn.benchmark = True

    ################# 出力フォルダ作成 ####################
    if not os.path.exists("{}".format(args.out)):
      	    os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "model")):
      	    os.mkdir(os.path.join("{}".format(args.out), "model"))
    if not os.path.exists(os.path.join("{}".format(args.out), "image")):
      	    os.mkdir(os.path.join("{}".format(args.out), "image"))
    if not os.path.exists(os.path.join("{}".format(args.out), "txt")):
      	    os.mkdir(os.path.join("{}".format(args.out), "txt"))
    PATH_1 = "{}/txt/trainloss.txt".format(args.out)
    PATH_2 = "{}/txt/testloss.txt".format(args.out)
    PATH_3 = "{}/txt/IoU.txt".format(args.out)
    PATH_4 = "{}/txt/IoU_test.txt".format(args.out)
    PATH_5 = "{}/txt/IoU_test_attention.txt".format(args.out)
    PATH_mse = "{}/txt/iou_attention.txt".format(args.out)
    PATH_index = "{}/txt/index_test.txt".format(args.out)

    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass
    with open(PATH_4, mode = 'w') as f:
        pass
    with open(PATH_mse, mode = 'w') as f:
        pass

    ################ データ読み込み ###############
    # 初期値
    sample = 0
    sample2 = 0
    tsample = 0
    n_class = 5
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    # データローダー作成
    ds_train = FlyCellDataLoader_crossval(rootdir=args.rootdir, val_area=args.val_area, split='train', iteration_number=args.batchsize*args.iter)
    ds_val = FlyCellDataLoader_crossval(rootdir=args.rootdir, val_area=args.val_area, split='val')
    ds_test = FlyCellDataLoader_crossval(rootdir=args.rootdir, val_area=args.val_area, split='test')
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batchsize, shuffle=True, num_workers=args.threads)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=8 , shuffle=False, num_workers=args.threads)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=args.threads)

    ################ モデル読み込み ###############
    model = UNet()

    model_u = smp.Unet(   encoder_name="se_resnet152" , 
        encoder_depth = 4,# choose encoder, e.g. mobilenet_v2 or efficientnet-b
        encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=5,                      # model output channels (number of classes in your dataset)
        decoder_channels = (256,128,64,32) 
    )

    model_dict = torch.load('load_model/model_bestiou_SEResnet.pth',map_location=device)
    model_u.load_state_dict(model_dict)

    if args.cuda:
        model = model.to(device) #モデルをGPUモードに
        model_u = model_u.to(device) #モデルをGPUモードに

    ################ class weight　作成 ###############
    ds_W = FlyCellDataLoader_crossval(rootdir=args.rootdir, val_area=args.val_area, split='train', iteration_number=args.batchsize*args.iter)
    train_loader_w = torch.utils.data.DataLoader(ds_W, batch_size=1, shuffle=False, num_workers=args.threads)
    weight = class_weight(1,train_loader_w)
    print(weight)
    w = torch.from_numpy(weight).to(device)
    
    ################ オプティマイザー　設定 ###############    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
    

    ################ 学習+検証スタート ####################
    print('# GPU : {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.num_epochs))
    print('')
    aaa = 0
    lr = 0.0005
    gamma = 0

    # 学習＋検証ループ
    for epoch in range(args.num_epochs):
        epoch = epoch
        start_time = time.clock()
        gamma = gamma + 1/args.num_epochs
        # 学習関数
        loss_train,mse,loss_mse_0 , loss_mse_1 ,loss_mse_2, loss_mse_3, loss_mse_4 ,index_len = train(epoch,gamma,model,train_loader,w,args,device,optimizer,model_u)
        # 検証関数
        loss_test, mm ,m1,m2,m3,m4, m5,index_len,mm2 ,m21,m22,m23,m24, m25= val(epoch,test_loader,model,args,device,n_class,model_u)
        print(mm,sample)

        # test関数
        if mm > sample:
            tmm ,tm1,tm2,tm3,tm4,tm5,index_base= test(epoch,test_loader,model,args,device,n_class,model_u)


        end_time = time.clock()

        # 出力結果を書き込み
        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_train))
        with open(PATH_2, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_test))
        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch+1, mm,m1,m2,m3,m4,m5,index_len))
        with open(PATH_mse, mode = 'a') as f:
            f.write("\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch+1, mm2,m21,m22,m23,m24,m25,index_len))

        # 最もVALのmIOUが高いモデルの保存
        if mm > sample:
           sample = mm
           PATH_best ="{}/model/model_bestiou.pth".format(args.out)
           torch.save(model.state_dict(), PATH_best)

        if mm2 > sample2:
           sample2 = mm2
           PATH_best ="{}/model/model_bestiou_attention.pth".format(args.out)
           torch.save(model.state_dict(), PATH_best)

        #　最もTestのｍIOUが高いモデルの保存   
        if tmm > tsample:
            tsample = tmm
            PATH_best ="{}/model/model_bestiou_test.pth".format(args.out)
            torch.save(model.state_dict(), PATH_best)
            with open(PATH_4, mode = 'a') as f:
                f.write("\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epoch+1, tmm,tm1,tm2,tm3,tm4,tm5))
 

        ############ 結果表示 ##############
        print("epoch %d / %d" % (epoch+1, args.num_epochs))
        print('train Loss: %.4f' % loss_train)
        print('test Loss : %.4f' % loss_test)
        print("   mIoU   : %.4f" % mm)
        print("   tmIoU   : %.4f" % tmm)
        print("   attentionmax   : %.4f" % sample2)
        print("   tmIoUmax   : %.4f" % tsample)
        print("time = %f" % (end_time - start_time))
        print("Learning Late = %f" % (lr))
        print("")

