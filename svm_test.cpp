#include <iostream>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <string.h>
#include <dirent.h>

using namespace std; 
using namespace cv; 

void getFiles( string path, vector<string>& files ); 

int main() 
{ 
    for (int num = 0; num < 2; num ++) 
    {
        int response;
        int result = 0; 
        float accuracy;
        string numpath = "/home/zhoujie/liveness detection/svm/data/test_image/";
        char char_num[2];
        sprintf(char_num,"%d",num);
        string str_num = char_num;
        string str = numpath + str_num;
        const char* filePath = str.data();
        string base;
        vector<string> files; 
        getFiles(filePath, files ); 
        int number = files.size(); 
        cout <<"文件夹"<< num <<" 共有测试图片 " <<number <<" 张"<< endl;

        Ptr<ml::SVM>svm = ml::SVM::load("svm.xml");
    
        for (int i = 0;i < number;i++) 
        { 
            base = str + "/" + files[i];
            Mat inMat = imread(base.c_str()); 
            Mat p = inMat.reshape(1, 1); 
            p.convertTo(p, CV_32FC1); 
            response = (int)svm->predict(p);  // 核心代码，将检测的图片的标签返回回来，结果保存在response中
            // cout << "识别的数字为：" << response << endl;
            if (response == num) 
            { 
                result++; 
            } 
	 // else
	 // {
	 //	 cout << base.c_str() << " ERROR ! " << endl;
         // }
         } 
        accuracy = result*1.0/number;
        cout << "识别正确 " << result <<" 张，准确率： "<< accuracy << endl;
    }

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
           { 
		continue; 
	   }
        else if(ptr->d_type == 4) ///dir 
           { 
                files.push_back(ptr->d_name); 
           } 
    } 
    closedir(dir);
    sort(files.begin(), files.end());
}
