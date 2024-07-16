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

    float union_area = static_cast<float>(countNonZero(union_class_mask));
    if (union_area == 0)
        return 1;
    float intersection_area = static_cast<float>(countNonZero(intersection_class_mask));
    return  intersection_area / union_area;
}

float performance_measurement::evaluate_playing_field_segmentation(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask)
{
    float iou_background = evaluate_class(found_mask, ground_truth_mask, segmentation_label::background);
    float iou_cue = evaluate_class(found_mask, ground_truth_mask, segmentation_label::cue);
    float iou_black = evaluate_class(found_mask, ground_truth_mask, segmentation_label::black);
    float iou_solids = evaluate_class(found_mask, ground_truth_mask, segmentation_label::solids);
    float iou_stripes = evaluate_class(found_mask, ground_truth_mask, segmentation_label::stripes);
    float iou_playing_field = evaluate_class(found_mask, ground_truth_mask, segmentation_label::playing_field);

    cout << "background iou: " << iou_background << endl;
    cout << "cue iou: " << iou_cue << endl;
    cout << "black iou: " << iou_black << endl;
    cout << "solids iou: " << iou_solids << endl;
    cout << "stripes iou: " << iou_stripes << endl;
    cout << "playing iou: " << iou_playing_field << endl;
    cout << "mean iou: " << (iou_background + iou_cue + iou_black + iou_solids + iou_stripes + iou_playing_field) / 6 << endl; 
    cout << "--------------------------" << endl;

    return 0;
}