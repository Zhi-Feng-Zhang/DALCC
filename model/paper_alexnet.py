#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
#import  alexnet,Non_local
from model import  alexnet,Non_local
class DeviceChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DeviceChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_planes, 3, kernel_size=6, stride=1, padding=3)
        self.device_fc1 = nn.Conv2d(11, int(in_planes * out_planes), kernel_size=1, bias=False, padding=0)
        self.device_fc2 = nn.Conv2d(11, int(out_planes * out_planes), kernel_size=1, bias=False, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, device_id):
        feat = self.avg_pool(x)  # B in 1 1
        bn, c, _, _ = feat.shape  # B in
        device_id = torch.unsqueeze(torch.unsqueeze(device_id, -1), -1)  # B*12*1*1
        conv_param1 = self.device_fc1(device_id).view((bn, c, -1))  # B in*out
        feat = feat.view(bn, 1, c)  # B 1 in
        feat = torch.matmul(feat, conv_param1)  # B 1 out
        feat = self.leaky_relu(feat)  # B 1 out

        bn, _, c2 = feat.shape  ##B 1 out
        conv_param2 = self.device_fc2(device_id).view((bn, c2, c2))  # B out out
        feat = torch.matmul(feat, conv_param2)  # B 1 out

        gamma = self.sigmoid(feat.view((bn, c2, 1, 1)))  # B out 1 1

        gamma=gamma.view(bn,3,3) #B*3*3
        #gamma_norm = gamma / (torch.sum(gamma, 1, keepdim=True) + 1e-9) + 1e-9

        out = self.conv1(x) #B out H W
        return self.relu(out),gamma#gamma_norm

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
        

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()


        net = alexnet.alexnet(pretrained=True)
        self.features = nn.Sequential(*list(net.children())[0][:15])


        self.illum = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.dca = DeviceChannelAttention(64, 9)

        self.non_local=Non_local.NONLocalBlock2D(256,sub_sample=False, bn_layer=False)


        self.domain_classifier = nn.ModuleList([nn.Sequential(
            nn.Linear(256*6*6, 2),
            nn.LogSoftmax(dim=1),) for _ in range(11)])

    

    def forward(self, input_data,index,alpha,k):
        feature = self.features(input_data)  
        feature1 = self.non_local(feature)   
        b, c, h, w = feature.shape
        feature = feature.view(-1, c * h * w)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier[k](reverse_feature)
        illum_output = self.illum(feature1)  
        patch_pred,CCM = self.dca(illum_output, index)
        pred = torch.nn.functional.normalize(torch.sum(torch.sum(patch_pred, 2), 2), dim=1)
        pred_orignal=pred.clone()
        pred=pred.unsqueeze(1)
        pred=torch.matmul(pred,CCM).squeeze(1)
        return domain_output,pred,pred_orignal


if __name__=='__main__':
        #from torchsummary import  summary
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        net = CNNModel().to(device)
        #summary(net,(3,256,256))
        print(net)
        input = torch.randn([4, 3, 256, 256]).to(device)
        idex=np.ones((4,11))
        idex=torch.from_numpy(idex).to(device).type(torch.cuda.FloatTensor)
        domain_output,pred,pred_orignal=net(input,idex,0,0)
        print(domain_output)                #4 3
        print(pred)