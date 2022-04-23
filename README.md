[![License](https://img.shields.io/bower/l/MI)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

#  Official codes for 'Domain Adversarial Learning for Color Constancy'.

Code will be  available after the IJCAI-2022 meeting.

## 1: Required Environment Installation
Please use the command  `pip install -r requirements.txt` or manually install the following packages.
  + torch==1.9.0+cu111
  + numpy==1.20.1
  + matplotlib==3.3.4
  + opencv_python==4.5.3.56
  + torchsummary==1.5.1
  + scipy==1.6.2
  + torchvision==0.10.0+cu111

## 2: Data Preprocessing

For details on RAW image preprocessing principle, please refer to  [RAW_preprocessing_MATLAB](https://ridiqulous.com/process-raw-data-using-matlab-and-dcraw/comment-page-3/#comments/) and [RAW_preprocessing_Python](https://nbviewer.org/github/yourwanghao/CMUComputationalPhotography/blob/master/class2/notebook2.ipynb/).  


### Color Checker Data preprocessing:
+ Download the dataset on: https://www2.cs.sfu.ca/~colour/data/shi_gehler/
+ Store the download file in: `/dataset/colorconstancy/colorchecker2010/`
+ Set the path of the output file, such as: `/home/*/data/CC_full_size/` or `/home/*/data/CC_resize/`
+ Run the `./dataset/color_constancy_data_process_all.py` code

### Cube+  Data preprocessing
+ Download the dataset on: https://ipg.fer.hr/ipg/resources/color_constancy
+ Store the download file in: `/dataset/colorconstancy/Cube/`
+ Set the path of the output file, such as: `/home/*/data/Cube_full_size/` or `/home/*/data/Cube_resize/`
+ Run the `./dataset/color_constancy_data_process_all.py` code



### NUS Data preprocessing
+ Download the dataset on: https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html
+ Store the download file in: `/dataset/colorconstancy/NUS/`
+ Set the path of the output file, such as: `/home/*/data/NUS_full_size/` or `/home/*/data/NUS_resize/`
+ Run the `./dataset/color_constancy_data_process_all.py` code


### Summary
 + If you have done the above, run the `./dataset/data_read.py`  for a simple test.
 + Run `./dataset/data_loder.py` to check that the data loader is running correctly.
 + Refer to the `./dataset/fold_dalcc.py` to check the three cross-validation. The fold for cross-validation can be found in `./data_fold/`


## 3: Network Model Checking

  + Download the Alexnet model parameter  that pre-trained on ImageNet, and  run the `./model/alexnet.py` to check the AlexNet model.
  + Run the `./model/DALCC.py` to check the DALCC model.
 

## 4: Train and Test
  + Run the `./train.py` to train the DALCC model.
  + Run the `./test.py` to test the DALCC model.

## 5: Pretrained model
 + You can find the pretrained model on Cube+ and NUS-8 on [Pretrained Model](https://github.com/Zhi-Feng-Zhang/DALCC/). 









