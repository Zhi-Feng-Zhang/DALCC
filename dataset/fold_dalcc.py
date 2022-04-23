#!/usr/bin/env python
# coding=utf-8
"""
Three-fold cross-validation, the corresponding data split can be obtained from ./data_fold
"""
import torch
import dataset.data_loader as data_loader

flag=1  #Controls the selection of test and training sets.

if flag==1:
    print("Flag=",flag)
    #CC_0,1
    Canon1D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold3.txt"
    Canon5D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold3.txt"
    #NUS_2-9
    Canon1DsMkIII_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold3.txt"
    Canon600D_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold3.txt"
    FujifilmXM1_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold3.txt"
    NikonD5200_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold3.txt"
    OlympusEPL6_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold3.txt"
    PanasonicGX1_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold3.txt"
    SamsungNX2000_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold3.txt"
    SonyA57_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold3.txt"
    #Cube_10
    Canon550D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold3.txt"


    #CC_0,1

    Canon1D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold2.txt"
    Canon5D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold2.txt"
    #NUS_2-9
    Canon1DsMkIII_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold2.txt"
    Canon600D_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold2.txt"
    FujifilmXM1_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold2.txt"
    NikonD5200_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold2.txt"
    OlympusEPL6_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold2.txt"
    PanasonicGX1_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold2.txt"
    SamsungNX2000_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold2.txt"
    SonyA57_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold2.txt"
    #Cube_10
    Canon550D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold2.txt"



    #CC_0,1
    Canon1D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold1.txt"
    Canon5D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold1.txt"
    #NUS_2-9
    Canon1DsMkIII_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold1.txt"
    Canon600D_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold1.txt"
    FujifilmXM1_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold1.txt"
    NikonD5200_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold1.txt"
    OlympusEPL6_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold1.txt"
    PanasonicGX1_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold1.txt"
    SamsungNX2000_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold1.txt"
    SonyA57_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold1.txt"
    #Cube_10
    Canon550D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold1.txt"

if flag==2:
    print("Flag=",flag)
    #CC_0,1
    Canon1D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold3.txt"
    Canon5D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold3.txt"
    #NUS_2-9
    Canon1DsMkIII_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold3.txt"
    Canon600D_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold3.txt"
    FujifilmXM1_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold3.txt"
    NikonD5200_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold3.txt"
    OlympusEPL6_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold3.txt"
    PanasonicGX1_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold3.txt"
    SamsungNX2000_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold3.txt"
    SonyA57_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold3.txt"
    #Cube_10
    Canon550D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold3.txt"


    #CC_0,1

    Canon1D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold1.txt"
    Canon5D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold1.txt"
    #NUS_2-9
    Canon1DsMkIII_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold1.txt"
    Canon600D_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold1.txt"
    FujifilmXM1_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold1.txt"
    NikonD5200_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold1.txt"
    OlympusEPL6_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold1.txt"
    PanasonicGX1_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold1.txt"
    SamsungNX2000_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold1.txt"
    SonyA57_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold1.txt"
    #Cube_10
    Canon550D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold1.txt"



    #CC_0,1
    Canon1D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold2.txt"
    Canon5D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold2.txt"
    #NUS_2-9
    Canon1DsMkIII_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold2.txt"
    Canon600D_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold2.txt"
    FujifilmXM1_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold2.txt"
    NikonD5200_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold2.txt"
    OlympusEPL6_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold2.txt"
    PanasonicGX1_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold2.txt"
    SamsungNX2000_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold2.txt"
    SonyA57_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold2.txt"
    #Cube_10
    Canon550D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold2.txt"

if flag==3:
    print("Flag=",flag)
    #CC_0,1
    Canon1D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold1.txt"
    Canon5D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold1.txt"
    #NUS_2-9
    Canon1DsMkIII_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold1.txt"
    Canon600D_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold1.txt"
    FujifilmXM1_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold1.txt"
    NikonD5200_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold1.txt"
    OlympusEPL6_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold1.txt"
    PanasonicGX1_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold1.txt"
    SamsungNX2000_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold1.txt"
    SonyA57_fold1="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold1.txt"
    #Cube_10
    Canon550D_fold1="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold1.txt"


    #CC_0,1

    Canon1D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold2.txt"
    Canon5D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold2.txt"
    #NUS_2-9
    Canon1DsMkIII_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold2.txt"
    Canon600D_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold2.txt"
    FujifilmXM1_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold2.txt"
    NikonD5200_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold2.txt"
    OlympusEPL6_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold2.txt"
    PanasonicGX1_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold2.txt"
    SamsungNX2000_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold2.txt"
    SonyA57_fold2="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold2.txt"
    #Cube_10
    Canon550D_fold2="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold2.txt"



    #CC_0,1
    Canon1D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon1D/fold3.txt"
    Canon5D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/CC/Canon5D/fold3.txt"
    #NUS_2-9
    Canon1DsMkIII_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_1D_mark3/fold3.txt"
    Canon600D_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/canon_eos_600D/fold3.txt"
    FujifilmXM1_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/fuji/fold3.txt"
    NikonD5200_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/nikonD5200/fold3.txt"
    OlympusEPL6_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/olympus/fold3.txt"
    PanasonicGX1_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/panasonic/fold3.txt"
    SamsungNX2000_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/samsung/fold3.txt"
    SonyA57_fold3="/dataset/colorconstancy/fast_data/data/nus/splits/sony/fold3.txt"
    #Cube_10
    Canon550D_fold3="/dataset/colorconstancy/fast_data/data/fold_last/Cube/fold3.txt"



Dataset_Canon1D = data_loader.dataset(Traing=True, file_path_list=[Canon1D_fold1, Canon1D_fold2])
loader_Canon1D = torch.utils.data.DataLoader(Dataset_Canon1D, batch_size=8, shuffle=True,
                                             num_workers=12)
Dataset_Canon5D = data_loader.dataset(Traing=True, file_path_list=[Canon5D_fold1, Canon5D_fold2])
loader_Canon5D = torch.utils.data.DataLoader(Dataset_Canon5D, batch_size=8, shuffle=True,
                                             num_workers=12)


Dataset_Canon1DsMkIII = data_loader.dataset(Traing=True, file_path_list=[Canon1DsMkIII_fold1, Canon1DsMkIII_fold2])
loader_Canon1DsMkIII = torch.utils.data.DataLoader(Dataset_Canon1DsMkIII, batch_size=8, shuffle=True,
                                                   num_workers=12)
Dataset_Canon600D = data_loader.dataset(Traing=True, file_path_list=[Canon600D_fold1, Canon600D_fold2])
loader_Canon600D = torch.utils.data.DataLoader(Dataset_Canon600D, batch_size=8, shuffle=True,
                                               num_workers=12)
Dataset_FujifilmXM1 = data_loader.dataset(Traing=True, file_path_list=[FujifilmXM1_fold1, FujifilmXM1_fold2])
loader_FujifilmXM1 = torch.utils.data.DataLoader(Dataset_FujifilmXM1, batch_size=8, shuffle=True,
                                                 num_workers=12)
Dataset_NikonD5200 = data_loader.dataset(Traing=True, file_path_list=[NikonD5200_fold1, NikonD5200_fold2])
loader_NikonD5200 = torch.utils.data.DataLoader(Dataset_NikonD5200, batch_size=8, shuffle=True,
                                                num_workers=12)
Dataset_OlympusEPL6 = data_loader.dataset(Traing=True, file_path_list=[OlympusEPL6_fold1, OlympusEPL6_fold2])
loader_OlympusEPL6 = torch.utils.data.DataLoader(Dataset_OlympusEPL6, batch_size=8, shuffle=True,
                                                 num_workers=12)

Dataset_PanasonicGX1 = data_loader.dataset(Traing=True, file_path_list=[PanasonicGX1_fold1, PanasonicGX1_fold2])
loader_PanasonicGX1 = torch.utils.data.DataLoader(Dataset_PanasonicGX1, batch_size=8, shuffle=True,
                                                  num_workers=12)

Dataset_SamsungNX2000 = data_loader.dataset(Traing=True, file_path_list=[SamsungNX2000_fold1, SamsungNX2000_fold2])
loader_SamsungNX2000 = torch.utils.data.DataLoader(Dataset_SamsungNX2000, batch_size=8, shuffle=True,
                                                   num_workers=12)

Dataset_SonyA57 = data_loader.dataset(Traing=True, file_path_list=[SonyA57_fold1, SonyA57_fold2])
loader_SonyA57 = torch.utils.data.DataLoader(Dataset_SonyA57, batch_size=8, shuffle=True,
                                             num_workers=12)

Dataset_Canon550D = data_loader.dataset(Traing=True, file_path_list=[Canon550D_fold1, Canon550D_fold2])
loader_Canon550D = torch.utils.data.DataLoader(Dataset_Canon550D, batch_size=8, shuffle=True,
                                               num_workers=12)





test_Dataset_Canon1D = data_loader.dataset(Traing=False, file_path_list=[Canon1D_fold3])
test_loader_Canon1D = torch.utils.data.DataLoader(test_Dataset_Canon1D, batch_size=1, shuffle=True,
                                                  num_workers=12)
test_Dataset_Canon5D = data_loader.dataset(Traing=False, file_path_list=[Canon5D_fold3])
test_loader_Canon5D = torch.utils.data.DataLoader(test_Dataset_Canon5D, batch_size=1, shuffle=True,
                                                  num_workers=12)


test_Dataset_Canon1DsMkIII = data_loader.dataset(Traing=False, file_path_list=[Canon1DsMkIII_fold3])
test_loader_Canon1DsMkIII = torch.utils.data.DataLoader(test_Dataset_Canon1DsMkIII, batch_size=1, shuffle=True,
                                                        num_workers=12)
test_Dataset_Canon600D = data_loader.dataset(Traing=False, file_path_list=[Canon600D_fold3])
test_loader_Canon600D = torch.utils.data.DataLoader(test_Dataset_Canon600D, batch_size=1, shuffle=True,
                                                    num_workers=12)

test_Dataset_FujifilmXM1 = data_loader.dataset(Traing=False, file_path_list=[FujifilmXM1_fold3])
test_loader_FujifilmXM1 = torch.utils.data.DataLoader(test_Dataset_FujifilmXM1, batch_size=1, shuffle=True,
                                                      num_workers=12)

test_Dataset_NikonD5200 = data_loader.dataset(Traing=False, file_path_list=[NikonD5200_fold3])
test_loader_NikonD5200 = torch.utils.data.DataLoader(test_Dataset_NikonD5200, batch_size=1, shuffle=True,
                                                     num_workers=12)
test_Dataset_OlympusEPL6 = data_loader.dataset(Traing=False, file_path_list=[OlympusEPL6_fold3])
test_loader_OlympusEPL6 = torch.utils.data.DataLoader(test_Dataset_OlympusEPL6, batch_size=1, shuffle=True,
                                                      num_workers=12)

test_Dataset_PanasonicGX1 = data_loader.dataset(Traing=False, file_path_list=[PanasonicGX1_fold3])
test_loader_PanasonicGX1 = torch.utils.data.DataLoader(test_Dataset_PanasonicGX1, batch_size=1, shuffle=True,
                                                       num_workers=12)

test_Dataset_SamsungNX2000 = data_loader.dataset(Traing=False, file_path_list=[SamsungNX2000_fold3])
test_loader_SamsungNX2000 = torch.utils.data.DataLoader(test_Dataset_SamsungNX2000, batch_size=1, shuffle=True,
                                                        num_workers=12)

test_Dataset_SonyA57 = data_loader.dataset(Traing=False, file_path_list=[SonyA57_fold3])
test_loader_SonyA57 = torch.utils.data.DataLoader(test_Dataset_SonyA57, batch_size=1, shuffle=True,
                                                  num_workers=12)

test_Dataset_Canon550D = data_loader.dataset(Traing=False, file_path_list=[Canon550D_fold3])
test_loader_Canon550D = torch.utils.data.DataLoader(test_Dataset_Canon550D, batch_size=1, shuffle=True,
                                                    num_workers=12)

def data_loader_train(flag):

    if flag==1:
        return loader_Canon1D
    if flag==2:
        return loader_Canon5D
    if flag==3:
        return loader_Canon1DsMkIII
    if flag==4:
        return loader_Canon600D
    if flag==5:
        return loader_FujifilmXM1
    if flag==6:
        return  loader_NikonD5200
    if flag==7:
        return  loader_OlympusEPL6
    if flag==8:
        return loader_PanasonicGX1
    if flag==9:
        return loader_SamsungNX2000
    if flag==10:
        return loader_SonyA57
    if flag==11:
        return loader_Canon550D


def data_loader_test(flag):

    if flag == 1:
        return test_loader_Canon1D
    if flag == 2:
        return test_loader_Canon5D
    if flag == 3:
        return test_loader_Canon1DsMkIII

    if flag == 4:
        return test_loader_Canon600D

    if flag== 5:
        return test_loader_FujifilmXM1

    if flag == 6:
        return test_loader_NikonD5200

    if flag == 7:
        return test_loader_OlympusEPL6

    if flag == 8:
        return test_loader_PanasonicGX1

    if flag==9:
        return test_loader_SamsungNX2000

    if flag==10:
        return test_loader_SonyA57

    if flag==11:
        return test_loader_Canon550D

