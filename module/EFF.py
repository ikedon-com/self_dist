# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
#from module.sync_batchnorm import SynchronizedBatchNorm2d
from module.BatchRenormalization2D import BatchRenormalization2D
from torch.nn import init
#from module.backbone import build_backbone
#from module.backbone2 import build_backbone2
#from module.ASPP import ASPP
from module.efficientnet_pytorch import EfficientNet

class BatchRenormalization2D(nn.Module):

	def __init__(self, num_features,  eps=1e-05, momentum=0.01, r_d_max_inc_step = 0.0001):
		super(BatchRenormalization2D, self).__init__()

		self.eps = eps
		self.momentum = torch.tensor( (momentum), requires_grad = False)

		self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
		self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

		self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
		self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False) 

		self.max_r_max = 3.0
		self.max_d_max = 5.0

		self.r_max_inc_step = r_d_max_inc_step
		self.d_max_inc_step = r_d_max_inc_step

		self.r_max = torch.tensor( (1.0), requires_grad = False)
		self.d_max = torch.tensor( (0.0), requires_grad = False)

	def forward(self, x):

		device = self.gamma.device

		batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
		batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

		self.running_avg_std = self.running_avg_std.to(device)
		self.running_avg_mean = self.running_avg_mean.to(device)
		self.momentum = self.momentum.to(device)

		self.r_max = self.r_max.to(device)
		self.d_max = self.d_max.to(device)


		if self.training:

			r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).to(device).data.to(device)
			d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).to(device).data.to(device)

			x = ((x - batch_ch_mean) * r )/ batch_ch_std + d
			x = self.gamma * x + self.beta

			if self.r_max < self.max_r_max:
				self.r_max += self.r_max_inc_step * x.shape[0]

			if self.d_max < self.max_d_max:
				self.d_max += self.d_max_inc_step * x.shape[0]

		else:

			x = (x - self.running_avg_mean) / self.running_avg_std
			x = self.gamma * x + self.beta

		self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
		self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)

		return x


class DBR(nn.Module):

	def __init__(self, in_ch, mid_ch, out_ch):
		super(DBR, self).__init__()
		self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2, 0)
		self.bnd = nn.BatchNorm2d(out_ch)
		self.conv1 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1)
		self.bn1 = nn.BatchNorm2d(out_ch)
		self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
		self.bn2 = nn.BatchNorm2d(out_ch)
		self.conv3 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
		self.bn3 = nn.BatchNorm2d(out_ch)



	def forward(self, x, skip=None):
		h = F.relu(self.bnd(self.deconv(x)))
		#h = self.se(h) + h
		if skip is None:
			h = F.relu(self.bn1(self.conv1(h)))
		else:
			#skip = self.se(skip) * skip
			#skip = self.ab(skip) * skip
			h = F.relu(self.bn1(self.conv1(torch.cat([h, skip], dim=1))))
		h = F.dropout(h, 0.25)
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.dropout(h, 0.25)
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.dropout(h, 0.25)

		return h
class DBR(nn.Module):

	def __init__(self, in_ch, mid_ch, out_ch):
		super(DBR, self).__init__()
		self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2, 0)
		self.bnd = nn.BatchNorm2d(out_ch)
		self.conv1 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1)
		self.bn1 = nn.BatchNorm2d(out_ch)
		self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
		self.bn2 = nn.BatchNorm2d(out_ch)
		self.conv3 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
		self.bn3 = nn.BatchNorm2d(out_ch)



	def forward(self, x, skip=None):
		h = F.relu(self.bnd(self.deconv(x)))
		#h = self.se(h) + h
		if skip is None:
			h = F.relu(self.bn1(self.conv1(h)))
		else:
			#skip = self.se(skip) * skip
			#skip = self.ab(skip) * skip
			h = F.relu(self.bn1(self.conv1(torch.cat([h, skip], dim=1))))
		h = F.dropout(h, 0.25)
		h = F.relu(self.bn2(self.conv2(h)))
		h = F.dropout(h, 0.25)
		h = F.relu(self.bn3(self.conv3(h)))
		h = F.dropout(h, 0.25)

		return h

class Efficient(nn.Module):
	def __init__(self, n_ch, n_cls):
		super(Efficient, self).__init__()
		self.d3 = DBR(200, 328, 256//n_ch)
		self.d2 = DBR(256, 168, 128//n_ch)
		self.d1 = DBR(128, 96, 64//n_ch)
		self.d0 = nn.Conv2d(64//n_ch, n_cls, 1, 1, 0)
		#self.d0 = nn.Conv2d(64//n_ch,n_cls,3,1,padding=1)
		self.conv = nn.Conv2d(200, 3, 1, 1, 0)
		self.conv2 = nn.Conv2d(72, 3, 1, 1, 0)
		self.conv_feature = nn.Conv2d(344, 3, kernel_size=3, padding=1)
		self.conv_input = nn.Conv2d(1, 3, kernel_size=1, padding=0)

		self.aconv = nn.Sequential(
				nn.Conv2d(3, 3, 1, 1, 0),
				nn.BatchNorm2d(3, momentum=0.9),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Sigmoid())

		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=8, padding=1)



		self.backbone_E = EfficientNet.from_pretrained('efficientnet-b6')
		self.fc = nn.Linear(196608,1000)
		self.avg_pool = nn.AdaptiveAvgPool2d(16)
		self.avg_pool2 = nn.AdaptiveAvgPool2d(32)



	def forward(self, x):
		#print(x.shape)
		x = self.conv_input(x)
		endpoints = self.backbone_E.extract_endpoints(x)
		#encoder
		h3 = endpoints['reduction_4']
		h2 = endpoints['reduction_3']
		h1 = endpoints['reduction_2']
		h0 = endpoints['reduction_1']

		#decoder
		h  = self.d3(h3, h2)
		h  = self.d2(h , h1)
		h  = self.d1(h , h0)
		h  = self.d0(h)
		#h  = F.softmax(h, dim=1)
		#print(h.shape)
	
		return h





