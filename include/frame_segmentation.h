#ifndef FRAME_SEGMENTATION_H
#define FRAME_SEGMENTATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

enum label_id
{
    background,
    cue,
    black,
    solids,
    stripes,
    playing_field
};

void get_colored_frame_segmentation(const cv::Mat& src, cv::Mat& dst, bool preserve_background);
void get_frame_segmentation(const cv::Mat &src, cv::Mat &dst);

#endif