#include<ros/ros.h> //ros标准库头文件
#include<iostream> //C++标准输入输出库
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
 
//OpenCV2标准头文件
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

//include relevant hpp files
#include "detector_test.hpp"
#include "image_converter.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // ros::init(argc, argv, "opencv_test"); 
    // if(argc != 2)
    // {
    //     cout<<"usage:rosrun opencv_test opencv_test_node <path of picture>"<<endl;
    // }  //图片路径最好为绝对路径

    // VideoCapture capture(0);
 
	// while (true)
	// {
	// 	Mat frame;
	// 	capture >> frame;
	// 	imshow("读取视频", frame);
	// 	waitKey(30);	//延时30
	// }

    // Mat image = cv::imread(argv[1]);
    // detect(image);

    ros::init(argc, argv, "image_converter");
    ImageConverter ic;

    ros::spin();
}

