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

class FlyCellDataLoader_crossval():
    def __init__ (self, rootdir="", val_area="1", split='train', iteration_number = None):
        self.split = split
        self.training = True if split=='train' else False

        filelist_train = []
        filelist_val = []
        filelist_test = []
        test_area = val_area + 1 if val_area != 5 else 1
        for i in range(1,6):
            dataset =  [os.path.join(f'{rootdir}/5-fold/Area_{i}/{data}') for data in os.listdir(f'{rootdir}/5-fold/Area_{i}')]
            if i == val_area:
                filelist_test = filelist_test + dataset
            elif i == test_area:
                filelist_val = filelist_val + dataset
            else:
                filelist_train = filelist_train + dataset
        if split == 'train':
            self.filelist = filelist_train 
        elif split == 'val':
            self.filelist = filelist_val
        elif split == 'test':
            self.filelist = filelist_test

        if self.training:
            self.number_of_run = 1
            self.iterations = iteration_number
        else:
            self.number_of_run = 16
            self.iterations = None

        print(f'val_area : {val_area} test_area : {test_area} ', end='')
        print(f"{split} files : {len(self.filelist)}")

    def __getitem__(self, index):

        if self.training:
            index = random.randint(0, len(self.filelist)-1)
            dataset = self.filelist[index]
        else:
            dataset = self.filelist[index//self.number_of_run]

        # load files
        filename_data = os.path.join(dataset)
        inputs = np.load(filename_data)

        #split feats labels
        if self.training:
            x = random.randint(0, inputs.shape[0] - 256)
            y = random.randint(0, inputs.shape[0] - 256)
        else:
            x = index%self.number_of_run//4 * 256
            y = index%self.number_of_run%4 * 256

        features = inputs[x:x+256,y:y+256,0:1].transpose(2,0,1).astype(np.float32)
        features /= 255.0
        labels = inputs[x:x+256,y:y+256,-1].astype(int)
        
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(labels).long()

        return fts, lbs

    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations
