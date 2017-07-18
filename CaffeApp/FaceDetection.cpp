//
//  FaceDetection.cpp
//  CaffeApp
//
//  Created by nice on 2017/3/8.
//  Copyright © 2017年 Takuya Matsuyama. All rights reserved.
//

#include <iostream>
#include "CascadeCNN.h"
using namespace std;
using namespace cv;

int test() {
    
    string mean_file = "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/imagenet_mean.binaryproto";
    
    vector<string> model_file = {
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/12c/deploy.prototxt",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/12cal/deploy.prototxt",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/24c/deploy.prototxt",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/24cal/deploy.prototxt",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/48c/deploy.prototxt",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/48cal/deploy.prototxt"
    };
    
    vector<string> trained_file = {
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/12c/12c.caffemodel",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/12cal/12cal.caffemodel",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/24c/24c.caffemodel",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/24cal/24cal.caffemodel",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/48c/48c.caffemodel",
        "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/Models/48cal/48cal.caffemodel"
    };
    
    vector<cv::Rect> rectangles;
    
    string img_path = "/Users/nice/Downloads/caffe-master/CaffeMac/CaffeMac/2.jpg";
    Mat img = imread(img_path);
    
    
    CascadeCNN cascadeCNN(model_file,trained_file,mean_file);
    
    //    cascadeCNN.timer_begin();
    //    cascadeCNN.detection_test(img, rectangles);
    cascadeCNN.detection(img, rectangles);
    //    cascadeCNN.timer_end();
    
    for(int i = 0; i < rectangles.size(); i++)
        rectangle(img, rectangles[i], Scalar(255, 0, 0));
    imshow("face", img);
    waitKey(0);
    
    return 0;
}
