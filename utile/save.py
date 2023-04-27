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
import os

def Save_image(cll, img, img2, ano, index, index_len, path,args,epoch):
    if not os.path.exists(os.path.join("{}/image".format(args.out), "{}".format(epoch))):
      	    os.mkdir(os.path.join("{}/image".format(args.out), "{}".format(epoch)))
    img = np.argmax(img,axis=1)
    img2 = np.argmax(img2, axis=1)
    
    img3 = img
    #print(img3.shape)


    cll = cll[0]*255.0
    cll = np.transpose(cll, (1,2,0))
    cll = cv2.cvtColor(np.uint8(cll), cv2.COLOR_BGR2RGB)
    img = img[0]
    img2 = img2[0]
    img3 = img3[0]
    ano = ano[0]
    dst1 = np.zeros((256,256,3))
    dst2 = np.zeros((256,256,3))
    dst3 = np.zeros((256,256,3))


    dst1[img==0] = [1.0,0.0,0.0]#red
    dst1[img==1] = [0.0,1.0,0.0]#green
    dst1[img==2] = [0.0,0.0,1.0]#blue    
    dst1[img==3] = [1.0,1.0,0.0]#yellow    
    dst1[img == 4] = [0.0, 0.0, 0.0]  #black
    
    dst3[img3==0] = [1.0,0.0,0.0]#red
    dst3[img3==1] = [0.0,1.0,0.0]#green
    dst3[img3==2] = [0.0,0.0,1.0]#blue    
    dst3[img3==3] = [1.0,1.0,0.0]#yellow    
    dst3[img3==4] = [0.0,0.0,0.0]#black   
    #dst3[img3==5] = [1.0,1.0,1.0]#black   

    dst2[ano==0] = [1.0,0.0,0.0]#red
    dst2[ano==1] = [0.0,1.0,0.0]#green
    dst2[ano==2] = [0.0,0.0,1.0]#blue    
    dst2[ano==3] = [1.0,1.0,0.0]#yellow    
    dst2[ano==4] = [0.0,0.0,0.0]#black   

    plt.figure(figsize=(15,15))
    row = 2
    col = 2

    plt.imshow(dst3)
    plt.savefig(path)
    plt.imsave(path,dst3) #　imsaveにするとメモリとかがつかないからよい

    plt.clf()
    plt.close()

def Save_image_attention(attention,ano, index,index_len,index_len_maxpath,args,epoch):

    ano = ano[0]
 

    plt.figure(figsize=(15,15))
    row = 2
    col = 2
    #print(attention.shape)


    for i in range(index_len_max):

        attentionS = attention[i]


        num = index[i]
        label = ano[num[1]][num[2]]
        if not os.path.exists(os.path.join("{}/image/attentionmap".format(args.out), "pixel_h{}_w{}_L{}".format(num[1], num[2], label))):
            os.mkdir(os.path.join("{}/image/attentionmap".format(args.out), "pixel_h{}_w{}_L{}".format(num[1], num[2], label)))
                    
        path2 = path +'/pixel_h{}_w{}_L{}/epoch{}.png'.format(num[1], num[2], label,epoch )
        #path2 = path + '{}{}_{}.png'.format(num[1], num[2], label )
        #dst3[attentionS == 5] = [1.0, 1.0, 1.0]  #black
        plt.imsave(path2,attentionS) #　imsaveにするとメモリとかがつかないからよい
    
    plt.clf()
    plt.close()

def Save_image2(cll, img, ano, path):
    img = np.argmax(img,axis=1)
    cll = cll[0]*255.0
    cll = np.transpose(cll, (1,2,0))
    cll = cv2.cvtColor(np.uint8(cll), cv2.COLOR_BGR2RGB)
    img = img[0]
    ano = ano[0]
    dst1 = np.zeros((256,256,3))
    dst2 = np.zeros((256,256,3))



    dst1[img==0] = [1.0,0.0,0.0]#red
    dst1[img==1] = [0.0,1.0,0.0]#green
    dst1[img==2] = [0.0,0.0,1.0]#blue    
    dst1[img==4] = [1.0,1.0,0.0]#yellow    
    dst1[img==3] = [0.0,0.0,0.0]#black   

    dst2[ano==0] = [1.0,0.0,0.0]#red
    dst2[ano==1] = [0.0,1.0,0.0]#green
    dst2[ano==2] = [0.0,0.0,1.0]#blue    
    dst2[ano==4] = [1.0,1.0,0.0]#yellow    
    dst2[ano==3] = [0.0,0.0,0.0]#black   


    plt.figure(figsize=(20,20))
    row = 3
    col = 6

    plt.subplot(row, col, 1)
    plt.title("Input")
    plt.imshow(cll)

    plt.subplot(row, col, 2)
    plt.title("Segmentation")
    plt.imshow(dst1)

    plt.subplot(row, col, 3)
    plt.title("Annotation")
    plt.imshow(dst2)

    plt.axis=('off')
    plt.savefig(path)
    plt.clf()
    plt.close()
