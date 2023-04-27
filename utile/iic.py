import sys
import torch
import torch.nn.functional as F

def compute_joint (out1,out2):

    out1 = out1.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w  3,1,32,32
    out2 = out2.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w  3,1,32,32
    #print(out1)
    #print(out2)

    p_i_j = F.conv2d(out1, weight=out2, padding=(0, 0)) #3,3,1,1

    #print(p_i_j)
    #print(p_i_j.shape)
    

    p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False) #3,3

    current_norm = float(p_i_j.sum())
    p_i_j = p_i_j / current_norm

    pij = (p_i_j + p_i_j.t() )/ 2.



    return pij


def iic(out1,out2,EPS=sys.float_info.epsilon):

    #print(out1)
    out1 =F.softmax(out1, dim=1)
    out2 =F.softmax(out2, dim=1)
    ta,k,h,w = out1.size()
    #print(k)
    #print(ta)
    p_i_j = compute_joint(out1,out2)  # torch.Size([10, 10]) #[3,3]
   # print(p_i_j)

    p_i = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
    p_j = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

    # 同時確率の分布表から、変換画像の10パターンをsumをして周辺化し、元画像だけの周辺確率の分布表を作る
    #p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)

    # 同時確率の分布表から、元画像の10パターンをsumをして周辺化し、変換画像だけの周辺確率の分布表を作る
    #p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)


    p_i_j = torch.where(p_i_j < EPS, torch.tensor(
        [EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)


    alpha = 1.0  # 論文や通常の相互情報量の計算はalphaは1です

    loss = -1*(p_i_j * (torch.log(p_i_j) - alpha *
                        torch.log(p_j) - alpha*torch.log(p_i))).sum()

    return loss
