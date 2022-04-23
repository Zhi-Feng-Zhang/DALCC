#!/usr/bin/env python
# coding=utf-8
import numpy as np
import math
import torch

def metric(errors):
    errors=sorted(errors)
    def g(f):
        return np.percentile(errors,f)
    mean=np.mean(errors)
    median=g(50)
    trimean=0.25*(g(25)+2*g(50)+g(75))
    best25=np.mean(errors[:int(0.25*len(errors))])
    worst25=np.mean(errors[int(0.75*len(errors)):])
    pct95=g(95)
    dict={"mean":mean,"median":median,"trimean":trimean,"best25":best25,"worst25":worst25,"pct95":pct95}
    return dict

def Cross_Loss(pred,GT,deviceid):
    loss=torch.nn.NLLLoss().cuda(device=deviceid)
    return  loss(pred.float(),GT.long())



def angle_loss(illum1,illum2):
    illum1=torch.nn.functional.normalize(illum1,dim=1,p=2)
    illum2=torch.nn.functional.normalize(illum2,dim=1,p=2)
    dot=torch.clamp(torch.sum(illum1*illum2,dim=1),-0.999999,0.999999)
    angle=torch.acos(dot)*(180/math.pi)
    return torch.mean(angle)





