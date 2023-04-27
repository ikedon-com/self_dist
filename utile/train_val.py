import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from module.attention import CAM_Module ,PAM_Module
from utile.save import Save_image,Save_image_attention
from utile.IoU import IoU
from utile.finepix import check_pix,pix_attention, random_pix
from utile.returnpix import return_pix
from tqdm import tqdm
import os

def train(epoch,gamma,model,train_loader,w,args,device,optimizer,model_u):
    # 学習モード
    model.train()
    model_u.eval()
    # 初期値
    self_loss_MSE = 0
    self_loss_CE = 0
    mode='train'
    sum_loss = 0
    loss_mse = 0
    self_loss_MSE = 0
    index_len = 0

    # 学習ループ
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        ###########　入力とラベルの変換　#############     
        model.zero_grad() 
        if args.cuda:
            inputs = inputs.to(device)
            targets = targets.to(device)

        targets = Variable(targets)
        targets = targets.long()
        inputs = inputs.to(device)
        inputs = Variable(inputs)

        ###########　教師モデルの出力　#############     
        with torch.no_grad():
            output_u= model_u(inputs)

        ###########　生徒モデルの出力　#############     
        output,self_teature,index_len = model(inputs,targets,epoch,mode,output_u)

        ##########　Loss計算　#############   
        self_loss_CE = F.cross_entropy(self_teature,targets,weight=w).to(device)
        self_loss_MSE =F.mse_loss(output,self_teature)
        loss_Seg = F.cross_entropy(output,targets,weight=w).to(device)


        loss =  self_loss_CE + loss_Seg + self_loss_MSE
        loss.backward()
        optimizer.step()
        
        print("")
        print("epoch:%d iter:%d "%(epoch,batch_idx))
        print("attentionmap Loss  : %f"%self_loss_MSE)
        print("attentionmap Loss_2  : %f"%self_loss_CE)
        print("index_len          : %f"%index_len)
        print("     seg_LOSS      : %f" % loss_Seg)
        print("")

    return sum_loss, loss, loss, loss, loss, loss, loss,index_len

######## 検証関数 #########
def val(epoch,val_loader,model,args,device,n_class,model_u):
    # テストモード
    model.eval()
    # 初期値
    predict = []
    predict2 = []
    answer = []
    labels = np.arange(n_class)
    sum_loss = 0
    index_len = 0
    mode='val'
    # 検証ループ
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):  
            if args.cuda:
                inputs = inputs.to(device)
                targets = targets.to(device)
            inputs = Variable(inputs)
            targets = Variable(targets)
            targets = targets.long()
            aaa=0
            output_u = model_u(inputs)
            output,output2,index,index_len = model(inputs, targets,epoch,mode,output_u)
            inputs=inputs.to(device)
            loss = F.cross_entropy(output, targets)
            sum_loss += loss.item()
            output = output.cpu().numpy()
            output2 = output2.cpu().numpy()
            targets = targets.cpu().numpy()
            inputs = inputs.cpu().numpy()
            predict.append(output)
            predict2.append(output2)
            answer.append(targets)

            if batch_idx == 0:
                Save_image(inputs, output, output, targets, index, index_len, "{}/image/{}/img_test{}.png".format(args.out, epoch, batch_idx + 1), args,epoch)

        # IoU測定
        iou = IoU(predict, answer, label=labels)
        miou = np.sum(iou) / n_class
        print(iou)
        iou2 = IoU(predict2, answer, label=labels)
        miou2 = np.sum(iou2) / n_class
        print(iou2)

    return sum_loss/(batch_idx+1), miou ,iou[0],iou[1],iou[2],iou[3],iou[4],index_len,miou2 ,iou2[0],iou2[1],iou2[2],iou2[3],iou2[4]
    
def test(epoch , test_loader,model,args,device,n_class,model_u):
    # テストモード
    model.eval()
    model_u.eval()
    index_len = 0
    # 初期値
    predict = []
    self_teature_predict = []
    answer = []
    labels = np.arange(n_class)
    sum_loss = 0
    gamma=1
    
    # 検証ループ
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            if args.cuda:
                inputs = inputs.to(device)
                targets = targets.to(device)
            targets = Variable(targets)
            targets = targets.long()
            inputs = inputs.to(device)
            inputs = Variable(inputs)

            mode='test'
            output_u = model_u(inputs)
            output,self_teature,index,index_len = model(inputs, targets,epoch,mode,output_u)

            loss = F.cross_entropy(output, targets)
            sum_loss += loss.item()

            # 評価用にTensor型→numpy型に変換
            output = output.cpu().numpy()
            self_teature = self_teature.cpu().numpy()
            targets = targets.cpu().numpy()
            inputs = inputs.cpu().numpy()

            # 出力溜め
            predict.append(output)
            self_teature_predict.append(self_teature)
            answer.append(targets)
            
            Save_image(inputs,  self_teature, output, targets, index, index_len, "{}/image/{}/img_test{}.png".format(args.out, epoch, batch_idx + 1), args,epoch)


        iou = IoU(self_teature_predict, answer, label=labels)
        print(iou)
        miou = np.sum(iou) / n_class
        print(miou)

        return miou ,iou[0],iou[1],iou[2],iou[3],iou[4],index_len


















