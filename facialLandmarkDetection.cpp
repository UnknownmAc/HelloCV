#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"


using namespace std;
using namespace cv;
using namespace cv::face;

void draw_subdiv(Mat &img, Subdiv2D& subdiv, Scalar delaunay_color)
{
    bool draw;
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    
    for(size_t i = 0; i < triangleList.size(); ++i)
    {
        Vec6f t = triangleList[i];
        
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
        // MY PIECE OF CODE
        draw=true;
        
        for(int i=0;i<3;i++){
            if(pt[i].x>img.cols||pt[i].y>img.rows||pt[i].x<0||pt[i].y<0)
                draw=false;
        }
        if (draw){
            line(img, pt[0], pt[1], delaunay_color, 1);
            line(img, pt[1], pt[2], delaunay_color, 1);
            line(img, pt[2], pt[0], delaunay_color, 1);
        }
        
        
    }
}

int detectFaceLandmarks()
{
    // Load Face Detector
    CascadeClassifier faceDetector("/Users/niskumar/Work/niskumar/HelloCV/haarcascade_frontalface_alt2.xml");

    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("/Users/niskumar/Work/niskumar/HelloCV/lbfmodel.yaml");

    // Set up webcam for video capture
    VideoCapture cam(0);
    
    // Variable to store a video frame and its grayscale 
    Mat frame, gray;
    
    // Read a frame
    while(cam.read(frame))
    {
      
      // Find face
      vector<Rect> faces;
      // Convert frame to grayscale because
      // faceDetector requires grayscale image.
      cvtColor(frame, gray, COLOR_BGR2GRAY);

      // Detect faces
      faceDetector.detectMultiScale(gray, faces);
      
      // Variable for landmarks. 
      // Landmarks for one face is a vector of points
      // There can be more than one face in the image. Hence, we 
      // use a vector of vector of points. 
      vector< vector<Point2f> > landmarks;
      
      // Run landmark detector
      bool success = facemark->fit(frame,faces,landmarks);
      
      if(success)
      {
        // If successful, render the landmarks on the face
        for(int i = 0; i < landmarks.size(); i++)
        {
          //drawLandmarks(frame, landmarks[i]);
        }
          
        if(faces.size() > 0)
        {
            // Rectangle to be used with Subdiv2D
            Size size = frame.size();
            Rect rect(0, 0, size.width, size.height);
            Subdiv2D subdiv(rect);
            // If successful, render the landmarks on the face
            for(int i = 0; i < landmarks.size(); i++)
            {
                for( vector<Point2f>::iterator it = landmarks[i].begin(); it != landmarks[i].end(); it++)
                    subdiv.insert(*it);
                break;
            }
            Scalar delaunay_color(255,0,0);
            draw_subdiv( frame, subdiv, delaunay_color );
        }
      }

      // Display results 
      imshow("Facial Landmark Detection", frame);
      // Exit loop if ESC is pressed
      if (waitKey(1) == 27) break;
      
    }
    return 0;
}
