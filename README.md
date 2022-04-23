#  Official codes for 'DALCC: Domain Adversarial Learning for Color Constancy'.

Code will be  available after the IJCAI-2022 meeting.

##Data Preprocessing

For details on RAW image preprocessing principle, please refer to  [RAW预处理MATLAB](https://ridiqulous.com/process-raw-data-using-matlab-and-dcraw/comment-page-3/#comments/) and [RAW预处理Python](https://nbviewer.org/github/yourwanghao/CMUComputationalPhotography/blob/master/class2/notebook2.ipynb/).  


###Color Checker Data preprocessing:
1: Download the dataset on: https://www2.cs.sfu.ca/~colour/data/shi_gehler/
2: Store the download file in: /dataset/colorconstancy/colorchecker2010/
3: Set the path of the output file, such as: /home/***/data/CC_full_size or /home/***/data/CC_resize
4: Run the ./dataset/color_constancy_data_process_all.py code

###Cube+ :Data preprocessing
1: Download the dataset on: https://ipg.fer.hr/ipg/resources/color_constancy
2: Store the download file in: /dataset/colorconstancy/Cube/
3: Set the path of the output file, such as: /home/***/data/Cube_full_size or /home/***/data/Cube_resize
4: Run the ./dataset/color_constancy_data_process_all.py code


"""
###NUS :Data preprocessing
1: Download the dataset on: https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html
2: Store the download file in: /dataset/colorconstancy/NUS/
3: Set the path of the output file, such as: /home/***/data/NUS_full_size or /home/***/data/NUS_resize
4: Run the ./dataset/color_constancy_data_process_all.py code
"""
