#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include "balls_localization.h"

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

struct match
{
    bool is_true_positive;
    float confidence;
};
typedef struct match match;

float evaluate_balls_localization(const balls_localization &localization, const balls_localization &ground_truth_localization);
float evaluate_balls_and_playing_field_segmentation(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask);
void frame_segmentation(const cv::Mat &src, cv::Mat &dst);
void load_ground_truth_localization(const std::string &filename, balls_localization &ground_truth_localization);

#endif