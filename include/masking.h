#ifndef MASK_H
#define MASK_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void mask_hsv(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask);
void mask_bgr(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mask);

#endif