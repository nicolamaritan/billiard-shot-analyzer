#ifndef FRAME_SEGMENTATION_H
#define FRAME_SEGMENTATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

enum segmentation_label
{
    background,
    cue,
    black,
    solids,
    stripes,
    playing_field
};

void frame_segmentation(const cv::Mat &src, cv::Mat &dst);

#endif