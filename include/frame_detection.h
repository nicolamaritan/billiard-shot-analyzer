#ifndef FRAME_DETECTION_H
#define FRAME_DETECTION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void get_frame_detection(const cv::Mat &src, cv::Mat &dst);

#endif