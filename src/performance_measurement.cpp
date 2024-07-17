#include "performance_measurement.h"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

float get_class_iou(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask, label_id class_id);
float get_class_iou(const ball_localization &localization, const ball_localization &ground_truth_localization);
float get_class_iou(const ball_localization &localization, const vector<ball_localization> &ground_truth_localizations);
match get_match(const ball_localization &predicted, label_id predicted_label, const balls_localization ground_truth);
float get_iou(const Mat &mask_1, const Mat &mask_2);
float get_iou(const Rect &rect_1, const Rect &rect_2);

void frame_segmentation(const Mat &src, Mat &dst)
{
    playing_field_localizer plf_localizer;
    plf_localizer.localize(src);
    playing_field_localization plf_localization = plf_localizer.get_localization();

    balls_localizer blls_localizer(plf_localization);
    blls_localizer.localize(src);
    balls_localization blls_localization = blls_localizer.get_localization();

    Mat segmentation(src.size(), CV_8UC1);
    segmentation.setTo(Scalar(label_id::background));
    segmentation.setTo(Scalar(label_id::playing_field), plf_localization.mask);

    Vec3f cue_circle = blls_localization.cue.circle;
    circle(segmentation, Point(cue_circle[0], cue_circle[1]), cue_circle[2], Scalar(label_id::cue), FILLED);

    Vec3f black_circle = blls_localization.black.circle;
    circle(segmentation, Point(black_circle[0], black_circle[1]), black_circle[2], Scalar(label_id::black), FILLED);

    for (ball_localization loc : blls_localization.solids)
    {
        Vec3f loc_circle = loc.circle;
        circle(segmentation, Point(loc_circle[0], loc_circle[1]), loc_circle[2], Scalar(label_id::solids), FILLED);
    }

    for (ball_localization loc : blls_localization.stripes)
    {
        Vec3f loc_circle = loc.circle;
        circle(segmentation, Point(loc_circle[0], loc_circle[1]), loc_circle[2], Scalar(label_id::stripes), FILLED);
    }

    dst = segmentation;
}

float get_class_iou(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask, label_id class_id)
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
    return intersection_area / union_area;
}

float get_iou(const Mat &mask_1, const Mat &mask_2)
{
    Mat union_mask;
    bitwise_or(mask_1, mask_2, union_mask);
    Mat intersection_mask;
    bitwise_and(mask_1, mask_2, intersection_mask);

    float union_area = static_cast<float>(countNonZero(union_mask));
    if (union_area == 0)
        return 1;
    float intersection_area = static_cast<float>(countNonZero(intersection_mask));
    return intersection_area / union_area;
}

float get_iou(const Rect &rect_1, const Rect &rect_2)
{
    Mat mask_1, mask_2;
    rectangle(mask_1, rect_1, cv::Scalar(255), FILLED);
    rectangle(mask_2, rect_2, cv::Scalar(255), FILLED);
    return get_iou(rect_1, rect_2);
}

float evaluate_balls_and_playing_field_segmentation(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask)
{
    float iou_background = get_class_iou(found_mask, ground_truth_mask, label_id::background);
    float iou_cue = get_class_iou(found_mask, ground_truth_mask, label_id::cue);
    float iou_black = get_class_iou(found_mask, ground_truth_mask, label_id::black);
    float iou_solids = get_class_iou(found_mask, ground_truth_mask, label_id::solids);
    float iou_stripes = get_class_iou(found_mask, ground_truth_mask, label_id::stripes);
    float iou_playing_field = get_class_iou(found_mask, ground_truth_mask, label_id::playing_field);

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

match get_match(const ball_localization &predicted, label_id predicted_label, const balls_localization ground_truth)
{
    float current_iou = 0;

    float max_iou = get_iou(predicted.bounding_box, ground_truth.cue.bounding_box);
    label_id max_iou_id = label_id::cue;
    float max_iou_confidence = 0;

    current_iou = get_iou(predicted.bounding_box, ground_truth.black.bounding_box);
    if (current_iou > max_iou)
    {
        max_iou = current_iou;
        max_iou_id = label_id::black;
        // TODO confidence
    }

    for (ball_localization solid_localization : ground_truth.solids)
    {
        current_iou = get_iou(predicted.bounding_box, solid_localization.bounding_box);
        if (current_iou > max_iou)
        {
            max_iou = current_iou;
            max_iou_id = label_id::solids;
            // TODO confidence
        }
    }

    for (ball_localization stripe_localization : ground_truth.stripes)
    {
        current_iou = get_iou(predicted.bounding_box, stripe_localization.bounding_box);
        if (current_iou > max_iou)
        {
            max_iou = current_iou;
            max_iou_id = label_id::solids;
            // TODO confidence
        }
    }

    const float IOU_THRESHOLD = 0.5;
    match returned_match;
    returned_match.is_true_positive = (predicted_label == max_iou_id && max_iou > IOU_THRESHOLD);
    returned_match.confidence = max_iou_confidence;

    return returned_match;
}

float evaluate_balls_localization(const balls_localization &predicted, const balls_localization &ground_truth)
{
    /*Mat cue_ground_truth_bbox_mask;
    rectangle(cue_ground_truth_bbox_mask, ground_truth.cue.bounding_box, cv::Scalar(255), FILLED);

    Mat black_ground_truth_bbox_mask;
    rectangle(black_ground_truth_bbox_mask, ground_truth.black.bounding_box, cv::Scalar(255), FILLED);

    vector<Mat> solids_ground_truth_bbox_masks;
    for (ball_localization solid_localization : ground_truth.solids)
    {
        Mat solid_bbox_mask;
        rectangle(solid_bbox_mask, solid_localization.bounding_box, cv::Scalar(255), FILLED);
        solids_ground_truth_bbox_masks.push_back(solid_bbox_mask);
    }

    vector<Mat> stripes_ground_truth_bbox_masks;
    for (ball_localization stripe_localization : ground_truth.stripes)
    {
        Mat stripe_bbox_mask;
        rectangle(stripe_bbox_mask, stripe_localization.bounding_box, cv::Scalar(255), FILLED);
        stripes_ground_truth_bbox_masks.push_back(stripe_bbox_mask);
    }*/

    match cue_match = get_match(predicted.cue, label_id::cue, ground_truth);
    // Compute precision and recall as follows since we are gauranteed to have one prediction and one ground truth for the cue
    float cue_precision = cue_match.is_true_positive ? 1 : 0;
    float cue_recall = cue_match.is_true_positive ? 1 : 0;

    match black_match = get_match(predicted.black, label_id::black, ground_truth);
    // As above for cue ball
    float black_precision = cue_match.is_true_positive ? 1 : 0;
    float black_recall = cue_match.is_true_positive ? 1 : 0;

    return 0;
}

void load_ground_truth_localization(const string &filename, balls_localization &ground_truth_localization)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Could not open the file!" << endl;
        return;
    }

    int x, y, width, height, label_id;
    while (file >> x >> y >> width >> height >> label_id)
    {
        ball_localization ball;
        ball.bounding_box = Rect(x, y, width, height);

        switch (label_id)
        {
        case label_id::cue:
            ground_truth_localization.cue = ball;
            break;
        case label_id::black:
            ground_truth_localization.black = ball;
            break;
        case label_id::solids:
            ground_truth_localization.solids.push_back(ball);
            break;
        case label_id::stripes:
            ground_truth_localization.stripes.push_back(ball);
            break;
        default:
            cerr << "Unknown label_id: " << label_id << endl;
            break;
        }
    }

    file.close();
}