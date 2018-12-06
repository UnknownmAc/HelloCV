//
//  main.cpp
//  HelloCV
//
//  Created by Huy Nguyen on 5/3/13.
//  Copyright (c) 2013 HelloCv. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <unistd.h>
#include "facialLandmarkDetection.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const double THRESHOLD = 400;
Mat baseImg, oldBase;
cv::Rect region1 = cv::Rect(60, 60, 320, 550);
cv::Rect region2 = cv::Rect(340, 40, 350, 580);
cv::Rect region3 = cv::Rect(680, 40, 295, 580);
cv::Rect region4 = cv::Rect(940, 40, 400, 580);

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        static int flag = 0;
        cv::Mat overlay;
        double alpha = 0.2;
        baseImg = oldBase.clone();
        // copy the source image to an overlay
        baseImg.copyTo(overlay);
        
        // draw a filled, yellow rectangle on the overlay copy
        if(flag%4 == 0)
            cv::rectangle(overlay, region1, cv::Scalar(0, 0, 255), -1);
        else if(flag%4 == 1)
            cv::rectangle(overlay, region2, cv::Scalar(0, 0, 255), -1);
        else if(flag%4 == 2)
            cv::rectangle(overlay, region3, cv::Scalar(0, 0, 255), -1);
        else if(flag%4 == 3)
            cv::rectangle(overlay, region4, cv::Scalar(0, 0, 255), -1);
        
        // blend the overlay with the source image
        cv::addWeighted(overlay, alpha, baseImg, 1 - alpha, 0, baseImg);
        
        imshow( "Base Image", baseImg );
        flag++;
    }
    else if  ( event == EVENT_RBUTTONDOWN )
    {
    }
    else if  ( event == EVENT_MBUTTONDOWN )
    {
    }
    else if ( event == EVENT_MOUSEMOVE )
    {
    }
}

int main(int argc, const char * argv[])
{
#if 0
    std::map<int, std::string> fileNameMap;
    fileNameMap[0] = "/Users/niskumar/Work/niskumar/PaintApp/object4.png";
    //fileNameMap[1] = "/Users/niskumar/Desktop/ImageStitching/gpart2.png";
    //fileNameMap[2] = "/Users/niskumar/Desktop/ImageStitching/gpart3.png";
    //fileNameMap[3] = "/Users/niskumar/Desktop/ImageStitching/gpart4.png";
    
    baseImg = imread("/Users/niskumar/Work/niskumar/PaintApp/object4.png", IMREAD_UNCHANGED );
    oldBase = baseImg.clone();
    namedWindow("Base Image", 1);
    //set the callback function for any mouse event
    setMouseCallback("Base Image", CallBackFunc, NULL);
    imshow( "Base Image", baseImg );
    
    for(int i=0; i < fileNameMap.size(); i++)
    {
        Mat input = imread( fileNameMap[i], IMREAD_UNCHANGED );
        //imshow( "InputImage" + std::to_string(i+1), input );

        Mat img_1 = imread( fileNameMap[i], IMREAD_GRAYSCALE );
        Mat img_2 = imread( "/Users/niskumar/Work/niskumar/PaintApp/object3.png", IMREAD_GRAYSCALE );
        
        if( !img_1.data || !img_2.data )
        { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
        
        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 800;
        
        Ptr<SURF> detector = SURF::create(minHessian);
        
        std::vector<KeyPoint> keypoints_1, keypoints_2;
        
        detector->detect( img_1, keypoints_1 );
        detector->detect( img_2, keypoints_2 );
        
        //-- Step 2: Calculate descriptors (feature vectors)
        Ptr<SURF> extractor = SURF::create();
        
        Mat descriptors_1, descriptors_2;
        
        extractor->compute( img_1, keypoints_1, descriptors_1 );
        extractor->compute( img_2, keypoints_2, descriptors_2 );
        
        //-- Step 3: Matching descriptor vectors using FLANN matcher
        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptors_1, descriptors_2, matches );
        
        double max_dist = 0; double min_dist = 100;
        
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        
        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );
        
        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.
        std::vector< DMatch > good_matches;
        
        for( int i = 0; i < descriptors_1.rows; i++ )
        { if( matches[i].distance <= max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
        }
        
        //-- Draw only "good" matches
        Mat img_matches;
        drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        //-- Show detected matches
        imshow( fileNameMap[i], img_matches );
        imwrite( "/Users/niskumar/Desktop/out.jpg", img_matches );
        
        for( int i = 0; i < (int)good_matches.size(); i++ )
        {
            printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
        }
        
        
       
// overlay over image
#if 0
        cv::Mat overlay;
        double alpha = 0.1;
        
        // copy the source image to an overlay
        baseImg.copyTo(overlay);
        
        // draw a filled, yellow rectangle on the overlay copy
        cv::rectangle(overlay, cv::Rect(950, 40, 388, 580), cv::Scalar(0, 0, 255), -1);
        
        // blend the overlay with the source image
        cv::addWeighted(overlay, alpha, baseImg, 1 - alpha, 0, baseImg);
#endif
        
        //imshow( fileNameMap[i], baseImg );
    }
    
    cv::Mat pic = cv::Mat::zeros(250,250,CV_8UC3);
    
    //for(int i=0; i < 1000; i++)
    {
        std::string progressText = "Stitching Images...";
        //std::cout << progressText << "\n";
        cv::putText(pic, progressText,cv::Point(50,50), CV_FONT_HERSHEY_SIMPLEX, 0.5,    cv::Scalar(255),1,8,false);
        //-- Show detected matches
        //imshow( "ProgressBar", pic );
        
        usleep(10000000);
    }
    
    Mat good_out = imread( "/Users/niskumar/Desktop/good_out.png", IMREAD_UNCHANGED );

    //imshow( "OutputWindow", good_out );
    
    waitKey(0);
#endif
    detectFaceLandmarks();
    
    return 0;
}

