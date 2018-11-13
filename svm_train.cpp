#include <iostream> 
#include <string.h>
#include<time.h>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
// #include <opencv2/core/core.hpp>  
#include <dirent.h>
// #include <io.h> windows下读取文件列表 

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

    /*
    // opencv2
    CvSVMParams SVM_params;  // 定义了一个结构体变量用来配置这些参数
    SVM_params.svm_type = CvSVM::C_SVC;  // SVM的类型：C_SVC表示SVM分类器，C_SVR表示SVM回归 
    SVM_params.kernel_type = CvSVM::LINEAR;  // 核函数类型 :线性核LINEAR，多项式核POLY，径向基核RBF，sigmoid核SIGMOID
    SVM_params.degree = 0; // 核函数中的参数degree,针对多项式核函数
    SVM_params.gamma = 1;  // 核函数中的参数gamma,针对多项式/RBF/SIGMOID核函数
    SVM_params.coef0 = 0;  // 核函数中的参数,针对多项式/SIGMOID核函数
    SVM_params.C = 1;  // SVM最优问题参数，设置C-SVC，EPS_SVR和NU_SVR的参数
    SVM_params.nu = 0;  // SVM最优问题参数，设置NU_SVC，ONE_CLASS 和NU_SVR的参数
    SVM_params.p = 0;  //SVM最优问题参数，设置EPS_SVR 中损失函数p的值
    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01); //迭代训练过程的中止条件：最大迭代次数，公差
    */
    // opencv 3.4.1
    // svm分类算法在opencv3中有了很大的变动，取消了CvSVMParams这个类，因此在参数设定上会有些改变。
    Ptr<SVM> svm = SVM::create();    //创建一个分类器
    svm->setType(SVM::C_SVC);    
    svm->setKernel(SVM::LINEAR); 
    // svm->setDegree(0);
    // svm->setGamma(1);
    // svm->setCoef0(0);
    // svm->setC(1);
    // svm->setNu(0);
    // svm->setP(0);
    // svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 0.01));
    
    //训练 
    /*
    //opencv2
    CvSVM svm; 
    svm.train(trainingData, classes, Mat(), Mat(), SVM_params); 
    */
    // opencv 3.4.1
    Ptr<TrainData> tData =TrainData::create(trainingData, ROW_SAMPLE, classes);
    cout << "SVM: start train ..." << endl;

    clock_t start,finish;
    double totaltime;
    start=clock();

    svm->trainAuto(tData);  
    // 或者使用
    // svm->train(trainingData, ROW_SAMPLE, classes);

    //保存模型 
    /*
    //opencv2
    svm.save("svm.xml"); 
    */
    // opencv 3.4.1
    svm->save("svm.xml");
    cout<<"SVM: TRAIN SUCCESS !"<<endl; 
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"TRAIN TIME : "<<totaltime<<" S ！"<<endl;
    // getchar(); 
    return 0; 
} 

/*
// windows下使用，io.h 头文件可能不兼容跨平台操作。在windows下这个头文件运行稳定，但是在linux下这个头文件不能正常运行。
void getFiles( string path, vector<string>& files ) 
{ 
    // 参考https://blog.csdn.net/overlord_bingo/article/details/69952795
    long hFile = 0;  //文件句柄 
    // struct _finddata_t fileinfo; //文件信息：unsigned attrib，time_t time_create，time_t time_access，time_t time_write，_fsize_t size，char name[_MAX_FNAME] 
    string p; 
    if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) != -1) 
    { 
        do 
        { 
            if((fileinfo.attrib & _A_SUBDIR))  //比较文件类型是否是文件夹
            { 
                if(strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0) 
                    getFiles( p.assign(path).append("\\").append(fileinfo.name), files ); 
            } 
            else 
            { 
                files.push_back(p.assign(path).append("\\").append(fileinfo.name) ); 
            } 
        }while(_findnext(hFile, &fileinfo) == 0); //寻找下一个，成功返回0，否则-1 循环读取路径下的文件

        _findclose(hFile); 
    } 
} 
*/

// ubuntu下使用
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
                // printf("%s\n",ptr->d_name); 
                files.push_back(ptr->d_name); 
            }
        else if(ptr->d_type == 10) ///link file 
            //printf("%s\n",ptr->d_name); 
            {continue; }
        else if(ptr->d_type == 4) ///dir 
        { 
            files.push_back(ptr->d_name); 
            /* 
            memset(base,'\0',sizeof(base)); 
            strcpy(base,basePath); 
            strcat(base,"/"); 
            strcat(base,ptr->d_nSame); 
            readFileList(base); */ 
        } 
    } 
    closedir(dir);
    sort(files.begin(), files.end());
}

// void get_1(Mat& trainingImages, vector<int>& trainingLabels) 
// { 
//     string str = "/home/zhoujie/cProject/svm_numbers/data/train_image/1";
//     const char* filePath = str.data();
//     string base;
//     vector<string> files;
//     getFiles(filePath, files); 
//     int number = files.size(); 
//     for (int i = 0;i < number;i++) 
//     { 
//         // cout << "*************************** n = " << i << " ************************************ "<< endl; 
//         base = str + "/" + files[i];
//         Mat SrcImage=imread(base.c_str()); 
//         SrcImage= SrcImage.reshape(1, 1); 
//         // cout << SrcImage << endl; 
//         trainingImages.push_back(SrcImage); 
//         trainingLabels.push_back(1); 
//     } 
// } 

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
