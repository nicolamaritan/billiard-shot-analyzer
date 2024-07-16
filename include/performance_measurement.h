#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include "frame_segmentation.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class performance_measurement
{
public:
    float evaluate_playing_field_segmentation(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask);

private:
    float evaluate_class(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask, segmentation_label class_id);
};

#endif