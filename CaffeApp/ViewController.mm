//
//  ViewController.m
//  CaffeApp
//
//  Created by Takuya Matsuyama on 7/11/15.
//  Copyright (c) 2015 Takuya Matsuyama. All rights reserved.
//

#import "ViewController.h"
#import "Classifier.h"

#include "CascadeCNN.h"
using namespace std;
using namespace cv;

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  // Do any additional setup after loading the view, typically from a nib.
  
//  UIImage* image = [UIImage imageNamed:@"img_4_4.png"];
    UIImage* image = [UIImage imageNamed:@"dog.jpg"];
  CGSize imageSize = image.size;
  CGSize viewSize = self.view.bounds.size;
  
  UIImageView* imageView = [[UIImageView alloc] initWithImage:image];
  imageView.frame = CGRectMake((viewSize.width-imageSize.width)/2, (viewSize.height-imageSize.height)/2, imageSize.width, imageSize.height);
  [self.view addSubview:imageView];

  UILabel* label = [[UILabel alloc] init];
  label.frame = CGRectMake(10, CGRectGetMaxY(imageView.frame)+10, viewSize.width-20, 50);
  label.numberOfLines = 2;
  label.textAlignment = NSTextAlignmentCenter;
  label.text = @"Classifying..";
  [self.view addSubview:label];
  
  dispatch_async(dispatch_get_main_queue(), ^{
//    NSString* result = [self predictWithImage:image];
//    label.text = result;
      [self detecFace:image];
  });
}

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
  // Dispose of any resources that can be recreated.
}

- (void)detecFace:(UIImage *)image
{

    string mean_file = [NSBundle.mainBundle pathForResource:@"imagenet_mean" ofType:@"binaryproto" inDirectory:@"model/face_model"].UTF8String;
    
    
    NSString* model1 = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model/face_model/12c"];
    NSString* model2 = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model/face_model/12cal"];
    NSString* model3 = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model/face_model/24c"];
    NSString* model4 = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model/face_model/24cal"];
    NSString* model5 = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model/face_model/48c"];
    NSString* model6 = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model/face_model/48cal"];
    vector<string> model_file;
    model_file.push_back(model1.UTF8String);
    model_file.push_back(model2.UTF8String);
    model_file.push_back(model3.UTF8String);
    model_file.push_back(model4.UTF8String);
    model_file.push_back(model5.UTF8String);
    model_file.push_back(model6.UTF8String);
    

    
    NSString* train1 = [NSBundle.mainBundle pathForResource:@"12c" ofType:@"caffemodel" inDirectory:@"model/face_model/12c"];
    NSString* train2 = [NSBundle.mainBundle pathForResource:@"12cal" ofType:@"caffemodel" inDirectory:@"model/face_model/12cal"];
    NSString* train3 = [NSBundle.mainBundle pathForResource:@"24c" ofType:@"caffemodel" inDirectory:@"model/face_model/24c"];
    NSString* train4 = [NSBundle.mainBundle pathForResource:@"24cal" ofType:@"caffemodel" inDirectory:@"model/face_model/24cal"];
    NSString* train5 = [NSBundle.mainBundle pathForResource:@"48c" ofType:@"caffemodel" inDirectory:@"model/face_model/48c"];
    NSString* train6 = [NSBundle.mainBundle pathForResource:@"48cal" ofType:@"caffemodel" inDirectory:@"model/face_model/48cal"];
    vector<string> trained_file;
    trained_file.push_back(train1.UTF8String);
    trained_file.push_back(train2.UTF8String);
    trained_file.push_back(train3.UTF8String);
    trained_file.push_back(train4.UTF8String);
    trained_file.push_back(train5.UTF8String);
    trained_file.push_back(train6.UTF8String);
    
    vector<cv::Rect> rectangles;
    
    
    
    Mat img;
    UIImageToMat(image, img);
    
    
    CascadeCNN cascadeCNN(model_file,trained_file,mean_file);
    
    //    cascadeCNN.timer_begin();
    //    cascadeCNN.detection_test(img, rectangles);
    
    const auto start = CACurrentMediaTime();
    
    for (int i=0; i<100; i++) {
        cascadeCNN.detection(img, rectangles);
    }

    NSLog(@"Time: %g seconds", CACurrentMediaTime() - start);
    
    for(int i = 0; i < rectangles.size(); i++)
        rectangle(img, rectangles[i], Scalar(255, 0, 0));
//    imshow("face", img);
//    waitKey(0);
}

- (NSString*)predictWithImage: (UIImage*)image;
{
//  NSString* model_file = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model"];
//  NSString* label_file = [NSBundle.mainBundle pathForResource:@"labels" ofType:@"txt" inDirectory:@"model"];
//  NSString* mean_file = [NSBundle.mainBundle pathForResource:@"mean" ofType:@"binaryproto" inDirectory:@"model"];
//  NSString* trained_file = [NSBundle.mainBundle pathForResource:@"bvlc_reference_caffenet" ofType:@"caffemodel" inDirectory:@"model"];
    NSString* model_file = [NSBundle.mainBundle pathForResource:@"lenet" ofType:@"prototxt" inDirectory:@"model"];
//    NSString* model_file = [NSBundle.mainBundle pathForResource:@"nin_imagenet_deploy" ofType:@"prototxt" inDirectory:@"model"];
//    NSString* model_file = [NSBundle.mainBundle pathForResource:@"nin_kaggle_deploy" ofType:@"prototxt" inDirectory:@"model"];
//    NSString* model_file = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model/Alex"];
//    NSString* model_file = [NSBundle.mainBundle pathForResource:@"deploy" ofType:@"prototxt" inDirectory:@"model/Inception"];
//    NSString* label_file = [NSBundle.mainBundle pathForResource:@"labels" ofType:@"txt" inDirectory:@"model/Alex"];
//    NSString* label_file = [NSBundle.mainBundle pathForResource:@"labels" ofType:@"txt" inDirectory:@"model"];
//    NSString* label_file = [NSBundle.mainBundle pathForResource:@"synset" ofType:@"txt" inDirectory:@"model/Inception"];
    NSString *label_file = @" ";
//    NSString* mean_file = [NSBundle.mainBundle pathForResource:@"mean" ofType:@"binaryproto" inDirectory:@"model"];
    NSString* mean_file = @" ";
    NSString* trained_file = [NSBundle.mainBundle pathForResource:@"lenet_iter_2000" ofType:@"caffemodel" inDirectory:@"model"];
//    NSString* trained_file = [NSBundle.mainBundle pathForResource:@"nin_imagenet_conv" ofType:@"caffemodel" inDirectory:@"model"];
//    NSString* trained_file = [NSBundle.mainBundle pathForResource:@"nin_kaggle_model" ofType:@"caffemodel" inDirectory:@"model"];
//    NSString* trained_file = [NSBundle.mainBundle pathForResource:@"bvlc_reference_caffenet" ofType:@"caffemodel" inDirectory:@"model/Alex"];
//    NSString* trained_file = [NSBundle.mainBundle pathForResource:@"Inception21k" ofType:@"caffemodel" inDirectory:@"model/Inception"];
  string model_file_str = std::string([model_file UTF8String]);
  string label_file_str = std::string([label_file UTF8String]);
  string trained_file_str = std::string([trained_file UTF8String]);
  string mean_file_str = std::string([mean_file UTF8String]);
  
  cv::Mat src_img, bgra_img;
  UIImageToMat(image, src_img);
  // needs to convert to BGRA because the image loaded from UIImage is in RGBA
//  cv::cvtColor(src_img, bgra_img, CV_RGBA2BGRA);
    bgra_img = src_img;

  Classifier classifier = Classifier(model_file_str, trained_file_str, mean_file_str, label_file_str);
    
    const auto start = CACurrentMediaTime();
    
    
//    for (int i=0; i<100; i++) {
      std::vector<Prediction> result = classifier.Classify(bgra_img);
//    }

    NSLog(@"Time: %g seconds", CACurrentMediaTime() - start);
    
  NSString* ret = nil;
  
  for (std::vector<Prediction>::iterator it = result.begin(); it != result.end(); ++it) {
    NSString* label = [NSString stringWithUTF8String:it->first.c_str()];
    NSNumber* probability = [NSNumber numberWithFloat:it->second];
    NSLog(@"label: %@, prob: %@", label, probability);
    if (it == result.begin()) {
      ret = label;
    }
  }
  
  return ret;
}

@end

