# FaceLivenessDetection_SVM
## 环境  
Ubuntu ，opencv3  
win、opencv2请参照： https://github.com/zj19941113/mnist_HandWriting_Recognition_SVM
## 数据获取  
使用MATLAB进行数据采集与处理  
* 运行dataGet_MATLAB/position_process.m，进行深度图片的人脸位置的快速批量标定  
* 运行dataGet_MATLAB/faceGet_process.m，进行人脸深度图的批量预处理  

![](https://github.com/zj19941113/FaceLivenessDetection_SVM/blob/master/img/face.png)  
![](https://github.com/zj19941113/FaceLivenessDetection_SVM/blob/master/img/noface.png)  
最终得到data.zip中的数据  
原始深度图 百度云盘：https://pan.baidu.com/s/1Hi85o521oIGaAfDoavOXeA  

## 模型训练
运行 svm_train.cpp  
>g++ svm_train.cpp \`pkg-config --cflags --libs opencv\` -o svm_train  
>./ svm_train 

## 分类器测试
运行 svm_test.cpp  
>g++ svm_test.cpp \`pkg-config --cflags --libs opencv\` -o svm_test  
>./ svm_test  

![](https://github.com/zj19941113/FaceLivenessDetection_SVM/blob/master/img/result.png)  
