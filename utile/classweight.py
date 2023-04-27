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
import random
from sklearn.metrics import confusion_matrix



######## 学習関数 ########
def class_weight(epoch,train_loader):
    a=0
    b=0
    c=0
    d=0
    e=0
    n_class=5
    w = np.zeros((n_class)).astype(np.float32)
    W = np.zeros((n_class)).astype(np.float32)
    W2 = np.zeros((5)).astype(np.float32)
    w2 = np.zeros((5)).astype(np.float32)

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        for j in range(n_class):
            w[j] = np.count_nonzero(targets == j)
        W[0] = w[0] + W[0]
        W[1] = w[1] + W[1]
        W[2] = w[2] + W[2]
        W[3] = w[3] + W[3]
        W[4] = w[4] + W[4]

    W2[0] = W[0]
    W2[1] = W[1]  
    W2[2] = W[2] 
    W2[3] = W[3] 
    W2[4] = W[4] 


    ww = np.median(W2).astype(np.float32)
    for j in range(5):
        if W2[j] != 0:
            w2[j] = ww/W2[j]
        else:
            w2[j]=0

    print(w2)

    
    return w2

def class_weight_4(epoch,train_loader):
    a=0
    b=0
    c=0
    d=0
    e=0
    n_class=4
    w = np.zeros((n_class)).astype(np.float32)
    W = np.zeros((n_class)).astype(np.float32)

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        for j in range(n_class):
            w[j] = np.count_nonzero(targets == j)
        W[0] = w[0] + W[0]
        W[1] = w[1] + W[1]
        W[2] = w[2] + W[2]
        W[3] = w[3] + W[3]
        #W[4] = w[4] + W[4]




    ww = np.median(W).astype(np.float32)
    for j in range(n_class):
        if W[j] != 0:
            w[j] = ww/W[j]
        else:
            w[j]=0
    """

    w[0] = 0.3115
    w[1] = 1.0000 
    w[2] = 12.0064
    w[3] = 2.0589
    w[4] = 0.0722

    print(W[0])
    print(W[1])
    print(W[2])
    print(W[3])
    print(W[4])
    print(w)
    """
    return w