#!/usr/bin/env python
# coding=utf-8
from model.Non_local import NONLocalBlock2D
import torch
import numpy as np
import os
#数据集
from dataset.fold_mdlcc import *
import model.paper_alexnet as fc4_model
import evaluation.metrics as metrics
import  random

Start=None
error_save_path=None
camere=11

def main():
    net = fc4_model.CNNModel()
    print(net)
    net.to("cuda:0")
    data = torch.load(Start)
    net.load_state_dict(data["model"])

    print("begin test*********************************************")
    dataloader_test = data_loader_test(camere)
    net.eval()
    val_errors = []
    with torch.no_grad():
        for images, illums, device_index in dataloader_test:
            images = images.to("cuda:0").type(torch.cuda.FloatTensor)
            illums = illums.to("cuda:0").type(torch.cuda.FloatTensor)
            device_index = device_index.to("cuda:0").type(torch.cuda.FloatTensor)
            domain_output, preds,pred_common = net(images,device_index, 0,0)
            loss = metrics.angle_loss(preds, illums)
            val_errors.append(loss.item())
                
        dict = metrics.metric(val_errors)
        print(dict)

if __name__ == "__main__":
   main()




