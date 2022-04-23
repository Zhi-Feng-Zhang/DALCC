#!/usr/bin/env python
# coding=utf-8
import numpy as np
import scipy.io
import os
import cv2
import matplotlib.pyplot as plt
#CC_1   CC_2   NUS1_ NUS2_ .... NUS_3   Cube_1
camera = ["Canon1D", "Canon5D", "Canon1DsMkIII", "Canon600D", "FujifilmXM1",
          "NikonD5200", "OlympusEPL6", "PanasonicGX1", "SamsungNX2000",
          "SonyA57", "Canon550D"]

def load_image(fn):
    if fn.startswith("IMG") or fn.startswith("8D5U"):
        img_path = "***/CC/" + fn.split(".")[0]
    else:
        if len(fn.split(".")[0])<5:
            img_path="***/Cube/"+fn.split(".")[0]
        else:
            img_path="***/NUS/"+fn.split(".")[0]

    img = np.load(img_path + ".npy").astype(np.float32)
    mask = np.load(img_path + "_mask.npy").astype(np.bool_)
    gt = np.load(img_path + "_gt.npy").astype(np.float32)
    camera_idx = np.load(img_path + "_camera.npy")
    

    index = camera.index(camera_idx)
    idx1, idx2, _ = np.where(mask == False)
    img[idx1, idx2, :] = 1e-5
    img[img == 0] = 1e-5
    C_index = np.zeros(11)
    C_index[index] = 1
    return img,gt ,C_index


if __name__ == '__main__':
    list=["Canon1DsMkIII_0002.PNG","1.PNG","8D5U5524.png"]
    for i in list:
        img ,gt,index= load_image(i)
        plt.imshow(np.power(img, 1 / 2.2))
        plt.show()
        print(gt)
        img=img/gt
        plt.imshow(np.power(img/img.max(), 1 / 2.2))
        plt.show()
        print(index)
        print(type(index))