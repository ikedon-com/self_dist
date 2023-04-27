import torch
import torch.nn as nn
import torch.nn.functional as F

#BatchRenormalization2Dはnn.BatchNorm2d()でも可

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

class CBR(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 2, 2, 0)
        self.bnc = BatchRenormalization2D(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.bn1 = BatchRenormalization2D(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, 1, 1)
        self.bn2 = BatchRenormalization2D(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1)
        self.bn3 = BatchRenormalization2D(out_ch)




    def forward(self, x, pool=False):
        if pool:
            h = F.relu(self.bnc(self.conv(x)))
            h = F.relu(self.bn1(self.conv1(h)))
        else:
            h = F.relu(self.bn1(self.conv1(x)))
        h = F.dropout(h, 0.25)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.dropout(h, 0.25)
        h = F.relu(self.bn3(self.conv3(h)))
        #ab = self.ab(h)
        h = F.dropout(h, 0.25)
        #h = self.se3(h) + h + h1 + h2

        return h


class DBR(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(DBR, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2, 0)
        self.bnd = BatchRenormalization2D(out_ch)
        self.conv1 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1)
        self.bn1 = BatchRenormalization2D(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = BatchRenormalization2D(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn3 = BatchRenormalization2D(out_ch)



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



class HiramatsuUNET(nn.Module):

    def __init__(self, n_ch, classes=12):
        super(HiramatsuUNET, self).__init__(
          
        )
        n_cls = classes
        self.c0 = CBR(1, 64//n_ch, 64//n_ch)
        self.c1 = CBR(64//n_ch, 128//n_ch, 128//n_ch)
        self.c2 = CBR(128//n_ch, 256//n_ch, 256//n_ch)
        self.c3 = CBR(256//n_ch, 512//n_ch, 512//n_ch)
        self.d3 = DBR(512//n_ch, 512//n_ch, 256//n_ch)
        self.d2 = DBR(256//n_ch, 256//n_ch, 128//n_ch)
        self.d1 = DBR(128//n_ch, 128//n_ch, 64//n_ch)
        self.d0 = nn.Conv2d(64//n_ch, classes, 1, 1, 0)
        self.fc = nn.Linear( 196608, 1000)#32*32*512 #32*32*200 204800
        self.conv = nn.Conv2d(512//n_ch,3,3,1,1)
        self.conv2 = nn.Conv2d(256//n_ch,3,3,1,1)
        self.conv_feature= nn.Conv2d(960,3,3,1,1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=8, padding=1)

            

    def forward(self, x):
        [b,c,row,col] = x.size()
        h0 = self.c0(x, False)
        h1 = self.c1(h0, True)
        h2 = self.c2(h1, True)
        h3 = self.c3(h2, True)

        h33 = F.interpolate(h3, (row,col), None, 'bilinear', True)
        h22 = F.interpolate(h2, (row,col), None, 'bilinear', True)
        h11 = F.interpolate(h1, (row,col), None, 'bilinear', True)
        feature_cat = torch.cat([h33,h22,h11,h0], dim=1)
        #print(feature_cat.shape)
        feature = self.conv_feature(feature_cat)

        feature2 = self.maxpool(feature)

        feature2 = feature2.view(feature2.size(0), -1)
        feature2 = F.softmax(feature2, dim=1)
        #print(feature2.shape)
        """
        h3_fc = self.conv(h3)
        h3_fc = h3_fc.view(h3_fc.size(0), -1) #h3_fc.size(0)=batch size
        #print(h3_fc.shape)
        h3_fc = F.softmax(h3_fc, dim=1)
        h2_fc = self.conv2(h2)
        h2_fc = h2_fc.view(h2_fc.size(0), -1) #h3_fc.size(0)=batch size
        #print(h3_fc.shape)
        h2_fc = F.softmax(h2_fc, dim=1)
        """

        
        h  = self.d3(h3, h2)
        h  = self.d2(h , h1)
        h  = self.d1(h , h0)
        #print(h.shape)
        h  = self.d0(h)
        


        

        return h

model = HiramatsuUNET(1, 3)
#print (model)

"""
UNET(
  (c0): CBR(
    (conv): Conv2d(1, 1, kernel_size=(2, 2), stride=(2, 2))
    (bnc): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (c1): CBR(
    (conv): Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
    (bnc): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (c2): CBR(
    (conv): Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
    (bnc): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (c3): CBR(
    (conv): Conv2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
    (bnc): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (d3): DBR(
    (deconv): ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
    (bnd): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (d2): DBR(
    (deconv): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
    (bnd): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (d1): DBR(
    (deconv): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
    (bnd): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (d0): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
)

"""