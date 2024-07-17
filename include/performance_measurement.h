#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include "balls_localization.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

enum match_type
{
    false_positive,
    true_positive
};

struct match
{
    match_type type;
    float confidence;
};
typedef struct match match;

float evaluate_balls_localization(const balls_localization &localization, const balls_localization &ground_truth_localization);
float evaluate_balls_and_playing_field_segmentation(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask);
void get_balls_localization(const cv::Mat &src, balls_localization &localization);
void load_ground_truth_localization(const std::string &filename, balls_localization &ground_truth_localization);

#endif