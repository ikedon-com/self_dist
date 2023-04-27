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


######## IoUの測定関数 ########
def IoU(output, target, label):
    # numpy型に変換
    output = np.array(output)
    target = np.array(target)

    # 確率最大値のラベルを出力
    seg = np.argmax(output,axis=2)

    # 1次元に
    seg = seg.flatten()
    target = target.flatten()

    # seg = ネットワークの予測
    # target = 正解ラベル 
    mat = confusion_matrix(target, seg, labels=label)
    mat = mat.astype(np.float32)
    mat_born = (mat - np.min(mat))/(np.max(mat)-np.min(mat))

    # ヒートマップ作成
    #sns.heatmap(mat_born, annot=True, fmt='1.2f', cmap='jet')
    #plt.savefig("{}/CM.png".format(args.out))

    # mIoU計算
    iou_den = (mat.sum(axis=1) + mat.sum(axis=0) - np.diag(mat))
    iou = np.array(np.diag(mat) ,dtype=np.float32) / np.array(iou_den, dtype=np.float32)

    return iou



