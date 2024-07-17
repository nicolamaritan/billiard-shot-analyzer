#ifndef PLAYING_FIELD_SEGMENTATION_H
#define PLAYING_FIELD_SEGMENTATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void playing_field_segmentation(const cv::Mat& src, cv::Mat& dst, bool preserve_background);

#endif