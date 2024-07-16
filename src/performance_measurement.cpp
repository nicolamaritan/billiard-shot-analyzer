#include "performance_measurement.h"
#include <iostream>

using namespace cv;
using namespace std;

float performance_measurement::evaluate_class(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask, segmentation_label class_id)
{
    CV_Assert(found_mask.channels() == 1);
    CV_Assert(ground_truth_mask.channels() == 1);

    Mat found_class_mask, ground_truth_class_mask;
    inRange(found_mask, Scalar(class_id), Scalar(class_id), found_class_mask);
    inRange(ground_truth_mask, Scalar(class_id), Scalar(class_id), ground_truth_class_mask);
    Mat union_class_mask, intersection_class_mask;
    bitwise_or(found_class_mask, ground_truth_class_mask, union_class_mask);
    bitwise_and(found_class_mask, ground_truth_class_mask, intersection_class_mask);
    return static_cast<float>(countNonZero(intersection_class_mask)) / static_cast<float>(countNonZero(union_class_mask));
}

float performance_measurement::evaluate_playing_field_segmentation(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask)
{
    cout << "background iou: " << evaluate_class(found_mask, ground_truth_mask, segmentation_label::background) << endl;
    cout << "cue iou: " << evaluate_class(found_mask, ground_truth_mask, segmentation_label::cue) << endl;
    cout << "black iou: " << evaluate_class(found_mask, ground_truth_mask, segmentation_label::black) << endl;
    cout << "solids iou: " << evaluate_class(found_mask, ground_truth_mask, segmentation_label::solids) << endl;
    cout << "stripes iou: " << evaluate_class(found_mask, ground_truth_mask, segmentation_label::stripes) << endl;
    cout << "playing iou: " << evaluate_class(found_mask, ground_truth_mask, segmentation_label::playing_field) << endl;
    cout << "--------------------------" << endl;

    return 0;
}