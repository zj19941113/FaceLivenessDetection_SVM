#include <iostream> 
#include <string.h>
#include<time.h>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <dirent.h>

using namespace std; 
using namespace cv;
using namespace cv::ml;

void getFiles( string path, vector<string>& files); 
void get_num(int num, Mat& trainingImages, vector<int>& trainingLabels); 

int main() 
{ 
    //获取训练数据 
    Mat classes; 
    Mat trainingData;
    Mat trainingImages; 
    vector<int> trainingLabels;
    get_num(0, trainingImages, trainingLabels);
    get_num(1, trainingImages, trainingLabels);
    
    Mat(trainingImages).copyTo(trainingData); 
    trainingData.convertTo(trainingData, CV_32FC1); 
    Mat(trainingLabels).copyTo(classes); 
	
    //配置SVM训练器参数 
    Ptr<SVM> svm = SVM::create();   
    svm->setType(SVM::C_SVC);    
    svm->setKernel(SVM::LINEAR); 
    
    Ptr<TrainData> tData =TrainData::create(trainingData, ROW_SAMPLE, classes);
    cout << "SVM: start train ..." << endl;

    clock_t start,finish;
    double totaltime;
    start=clock();

    svm->trainAuto(tData);  
    svm->save("svm.xml");
    cout<<"SVM: TRAIN SUCCESS !"<<endl; 
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"TRAIN TIME : "<<totaltime<<" S ！"<<endl;
    // getchar(); 
    return 0; 
} 

void getFiles( string path, vector<string>& files ) 
{ 
    DIR *dir;
	struct dirent *ptr;

    if ((dir=opendir(path.c_str())) == NULL)
    {
		perror("Open path error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL) 
    { 
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0) ///current dir OR parrent dir 
            continue; 
        else if(ptr->d_type == 8) ///file 
            {
                files.push_back(ptr->d_name); 
            }
        else if(ptr->d_type == 10) ///link file 
            {continue; }
        else if(ptr->d_type == 4) ///dir 
        { 
            files.push_back(ptr->d_name); 
        } 
    } 
    closedir(dir);
    sort(files.begin(), files.end());
}

void get_num(int num, Mat& trainingImages, vector<int>& trainingLabels) 
{ 
    string numpath = "/home/zhoujie/liveness detection/svm/data/train_image/";
    char char_num[2];
    sprintf(char_num,"%d",num);
    string str_num = char_num;
    string str = numpath + str_num;
    const char* filePath = str.data();
    string base;
    vector<string> files;
    getFiles(filePath, files); 
    int number = files.size(); 
    for (int i = 0;i < number;i++) 
    { 
        // cout << "*************************** n = " << i << " ************************************ "<< endl; 
        base = str + "/" + files[i];
        Mat SrcImage=imread(base.c_str()); 
        SrcImage= SrcImage.reshape(1, 1); 
        // cout << SrcImage << endl; 
        trainingImages.push_back(SrcImage); 
        trainingLabels.push_back(num); 
    } 
}
