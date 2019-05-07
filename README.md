# NCNN
A Non-sequential Convolutional Neural Network Model for Human Action Recognition in Still Images
############################ 1. Preface ########################################

Our paper: A Non-sequential Convolutional Neural Network Model for Human Action Recognition in Still Images

The code is provided by Xiangchun Yu.

Email:yxchspring0209@foxmail.com

Data: May 08, 2019

############################ 2. Data and Model ########################################

2.1 Please extract the 'Willow-actions.rar' to the 'NCNN' folder.

2.2 Please refer to the following link for the trained models including the best Model1 and the best Model2

Link: https://pan.baidu.com/s/1lVSsmLW_wYyy13EC6DR7Sg Extraction code: 17ju

Please extract the 'Results.rar' to the 'NCNN' folder.

2.3 We just provide the '1.5 × cropped Willow actions database' in this code.

And the original uncropped Willow actions database and code in Delaitre's Paper  are available at https://www.di.ens.fr/willow/research/stillactions/.

############################ 3. Functions ########################################

If the Keras and caret pacakages have been installed in your RStudio, the following codes can be run without manual adjustment. This program can be run under windows and macos systems (Linux will be OK, but we have not tested it.)

Model1.R: The code for the Model1

Model2.R: The code for the non-sequential CNN (NCNN)

Model3.R: The code for ensemble model between Model1 and Model2
