// Author: Nicola Maritan 2121717

#include "balls_localization.h"
#include "geometry.h"
#include "segmentation.h"

#include <opencv2/features2d.hpp>

#include <iostream>
#include <cmath>
#include <map>
#include <queue>
#include <cassert>

using namespace cv;
using namespace std;

bool operator==(const ball_localization &lhs, const ball_localization &rhs)
{
    return lhs.circle == rhs.circle && lhs.bounding_box == rhs.bounding_box;
}

bool operator!=(const ball_localization &lhs, const ball_localization &rhs)
{
    return !(lhs == rhs);
}

void balls_localizer::localize(const Mat &src)
{
    const int FILTER_SIZE = 3;
    const int FILTER_SIGMA = 3;
    Mat blurred;
    GaussianBlur(src, blurred, Size(FILTER_SIZE, FILTER_SIZE), FILTER_SIGMA, FILTER_SIGMA);

    Mat src_masked;
    src.copyTo(src_masked, playing_field.mask);

    Mat blurred_masked;
    blurred.copyTo(blurred_masked, playing_field.mask);

    Mat blurred_masked_hsv;
    cvtColor(blurred_masked, blurred_masked_hsv, COLOR_BGR2HSV);

    Mat src_masked_hsv;
    cvtColor(src_masked, src_masked_hsv, COLOR_BGR2HSV);

    // Computation of board, shadows and color masks
    Mat board_mask;
    Mat shadows_mask;
    Mat color_mask;
    Mat final_segmentation_mask;

    // Playing field color estimation
    int RADIUS = 100;
    const Vec3b board_color_hsv = get_playing_field_color(blurred_masked_hsv, RADIUS);
    const Vec3b SHADOW_OFFSET = Vec3b(0, 0, 90);

    Vec3b shadow_hsv = board_color_hsv - SHADOW_OFFSET;
    inRange(blurred_masked_hsv, board_color_hsv - Vec3b(5, 80, 50), board_color_hsv + Vec3b(5, 60, 15), board_mask);
    inRange(blurred_masked_hsv, shadow_hsv - Vec3b(3, 30, 80), shadow_hsv + Vec3b(3, 100, 40), shadows_mask);
    inRange(blurred_masked_hsv, board_color_hsv - Vec3b(10, 255, 150), shadow_hsv + Vec3b(10, 255, 255), color_mask);

    Mat outer_field;
    Mat shrinked_playing_field_mask;

    // Consider shadow mask only near the table edges
    const int DEPTH_SHADOW_MASK = 50;
    erode(playing_field.mask, shrinked_playing_field_mask, getStructuringElement(MORPH_CROSS, Size(DEPTH_SHADOW_MASK, DEPTH_SHADOW_MASK)));
    bitwise_not(shrinked_playing_field_mask, outer_field);
    bitwise_and(shadows_mask.clone(), outer_field, shadows_mask);

    // Consider color mask only near the table edges
    const int DEPTH_COLOR_MASK = 30;
    erode(playing_field.mask, shrinked_playing_field_mask, getStructuringElement(MORPH_CROSS, Size(DEPTH_COLOR_MASK, DEPTH_COLOR_MASK)));
    bitwise_not(shrinked_playing_field_mask, outer_field);
    bitwise_and(color_mask.clone(), outer_field, color_mask);

    // Union of the above masks
    bitwise_or(board_mask, shadows_mask, final_segmentation_mask);
    bitwise_or(final_segmentation_mask, color_mask, final_segmentation_mask);

    // Region growing to fine tune the mask
    const int HUE_THRESHOLD = 3;
    const int SATURATION_THRESHOLD = 6;
    const int VALUE_THRESHOLD = 4;
    vector<Point> seed_points;
    extract_seed_points(final_segmentation_mask, seed_points);
    region_growing(blurred_masked_hsv, final_segmentation_mask, seed_points, HUE_THRESHOLD, SATURATION_THRESHOLD, VALUE_THRESHOLD);

    // Closening operation to fine-tune the mask
    const Size CLOSURE_SIZE = Size(3, 3);
    morphologyEx(final_segmentation_mask.clone(), final_segmentation_mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, CLOSURE_SIZE));

    const int AREA_THRESHOLD = 90;
    fill_small_holes(final_segmentation_mask, AREA_THRESHOLD);

    // Remove black component outside the current masking. This is able to remove hands and some holes from the masking.
    Mat out_of_field_mask;
    mask_region_growing(final_segmentation_mask, out_of_field_mask, {Point(0, 0)});
    bitwise_or(final_segmentation_mask.clone(), out_of_field_mask, final_segmentation_mask);

    const int HOUGH_MIN_RADIUS = 8;
    const int HOUGH_MAX_RADIUS = 16;
    const float HOUGH_DP = 0.3;
    const int HOUGH_MIN_DISTANCE = 15;
    const int HOUGH_CANNY_PARAM = 100;
    const int HOUGH_MIN_VOTES = 5;
    vector<Vec3f> circles;
    HoughCircles(final_segmentation_mask, circles, HOUGH_GRADIENT, HOUGH_DP, HOUGH_MIN_DISTANCE, HOUGH_CANNY_PARAM, HOUGH_MIN_VOTES, HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS);

    vector<Mat> hough_circle_masks;
    circles_masks(circles, hough_circle_masks, src.size());

    // Circle filtering to remove wrongly detected circles by the transform.
    const float MAX_INTERSECTION = 0.60;
    const float MAX_DISTANCE_OUT_OF_BOUNDS = 20;
    const float MIN_DISTANCE_FROM_HOLE = 27;
    const float MIN_DISSIMILAR_NEIGHBORDHOOD_DISTANCE = 25;
    const float MIN_DISSIMILAR_VERTICAL_DISTANCE = 25;
    const float MIN_DISSIMILAR_RADIUS_DIFFERENCE = 2;
    filter_empty_circles(circles, hough_circle_masks, final_segmentation_mask, MAX_INTERSECTION);
    filter_out_of_bound_circles(circles, playing_field.mask, MAX_DISTANCE_OUT_OF_BOUNDS);
    filter_near_holes_circles(circles, playing_field.hole_points, MIN_DISTANCE_FROM_HOLE);
    filter_close_dissimilar_circles(circles, MIN_DISSIMILAR_NEIGHBORDHOOD_DISTANCE, MIN_DISSIMILAR_VERTICAL_DISTANCE, MIN_DISSIMILAR_RADIUS_DIFFERENCE);

    // Ball classification among detected circles
    find_cue_ball(blurred_masked, final_segmentation_mask, circles);
    find_black_ball(blurred_masked, final_segmentation_mask, circles);
    find_stripe_balls(src_masked, final_segmentation_mask, circles);
    find_solid_balls(blurred_masked, final_segmentation_mask, circles);

    get_bounding_boxes(circles, bounding_boxes);
}

void balls_localizer::circles_masks(const vector<Vec3f> &circles, vector<Mat> &masks, Size mask_size)
{
    masks.clear();
    for (size_t i = 0; i < circles.size(); i++)
    {
        Mat mask(mask_size, CV_8U);
        mask.setTo(Scalar(0));
        const Scalar WHITE = Scalar(255);
        circle(mask, Point(circles.at(i)[0], circles.at(i)[1]), circles.at(i)[2], WHITE, FILLED);
        masks.push_back(mask);
    }
}

void balls_localizer::filter_empty_circles(vector<Vec3f> &circles, const vector<Mat> &masks, const Mat &segmentation_mask, float intersection_threshold)
{
    CV_Assert(segmentation_mask.type() == CV_8UC1);

    vector<Vec3f> filtered_circles;
    for (size_t i = 0; i < circles.size(); i++)
    {
        /*
            Compute the ratio of the circle which is empty, i.e. that contains a large portion
            of segmentation mask. In fact, if a circle contains such large portion, it is probably been
            wrongly detected by Hough transform.
        */
        float circle_area = static_cast<float>(countNonZero(masks[i]));
        Mat masks_intersection;
        bitwise_and(masks[i], segmentation_mask, masks_intersection);
        float intersection_area = static_cast<float>(countNonZero(masks_intersection));

        if (intersection_area / circle_area < intersection_threshold)
            filtered_circles.push_back(circles.at(i));
    }
    circles = filtered_circles;
}

void balls_localizer::filter_out_of_bound_circles(vector<Vec3f> &circles, const Mat &table_mask, int distance_threshold)
{
    CV_Assert(table_mask.type() == CV_8UC1);

    vector<Vec3f> filtered_circles;
    Mat shrinked_table_mask;
    const int WHITE = 255;

    // Erode the table to exclude false positives in the table border
    erode(table_mask, shrinked_table_mask, getStructuringElement(MORPH_CROSS, Size(distance_threshold, distance_threshold)));

    for (Vec3f circle : circles)
    {
        Point center = Point(circle[0], circle[1]);
        // Keep the center only if it is inside the mask, i.e. not out of bounds
        if (shrinked_table_mask.at<uchar>(center) == WHITE)
            filtered_circles.push_back(circle);
    }
    circles = filtered_circles;
}

void balls_localizer::filter_near_holes_circles(vector<Vec3f> &circles, const vector<Point> &holes_points, float distance_threshold)
{
    vector<Vec3f> filtered_circles;
    for (const Vec3f circle : circles)
    {
        Point circle_point = Point(static_cast<int>(circle[0]), static_cast<int>(circle[1]));

        // Remove the circle if it is too close to the hole.
        bool keep_circle = true;
        for (Point hole_point : holes_points)
        {
            if (norm(hole_point - circle_point) < distance_threshold)
                keep_circle = false;
        }

        if (keep_circle)
            filtered_circles.push_back(circle);
    }
    circles = filtered_circles;
}

void balls_localizer::get_bounding_boxes(const vector<Vec3f> &circles, vector<Rect> &bounding_boxes)
{
    bounding_boxes.clear();

    for (const Vec3f circle : circles)
    {
        int center_x = static_cast<int>(circle[0]);
        int center_y = static_cast<int>(circle[1]);
        int radius = static_cast<int>(circle[2]);

        // Calculate top-left corner of the bounding box
        int x = center_x - radius;
        int y = center_y - radius;

        // Calculate width and height of the bounding box
        int width = 2 * radius;
        int height = 2 * radius;

        Rect bounding_box(x, y, width, height);
        bounding_boxes.push_back(bounding_box);
    }
}

cv::Rect balls_localizer::get_bounding_box(cv::Vec3f circle)
{
    int center_x = static_cast<int>(circle[0]);
    int center_y = static_cast<int>(circle[1]);
    int radius = static_cast<int>(circle[2]);

    // Calculate top-left corner of the bounding box
    int x = center_x - radius;
    int y = center_y - radius;

    // Calculate width and height of the bounding box
    int width = 2 * radius;
    int height = 2 * radius;

    return Rect(x, y, width, height);
}

void balls_localizer::fill_small_holes(Mat &binary_mask, double area_threshold)
{
    CV_Assert(binary_mask.type() == CV_8UC1);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary_mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    // Fills hole of components with area smaller than area_threshold
    for (int i = 0; i < contours.size(); ++i)
    {
        double area = contourArea(contours[i]);
        if (area < area_threshold)
            drawContours(binary_mask, contours, i, Scalar(255), FILLED, 8, hierarchy, 1); // Fills the hole
    }
}

void balls_localizer::extract_seed_points(const Mat &inrange_segmentation_mask, vector<Point> &seed_points)
{
    CV_Assert(inrange_segmentation_mask.type() == CV_8UC1);
    seed_points.clear();

    // Take as seed point every white pixel in the mask
    for (int y = 0; y < inrange_segmentation_mask.rows; ++y)
    {
        for (int x = 0; x < inrange_segmentation_mask.cols; ++x)
        {
            if (inrange_segmentation_mask.at<uchar>(y, x) > 0)
            {
                seed_points.push_back(Point(x, y));
            }
        }
    }
}

void balls_localizer::get_circle_and_field_mask(const cv::Mat &segmentation_mask, cv::Vec3f circle, cv::Mat &mask)
{
    CV_Assert(segmentation_mask.type() == CV_8UC1);
    CV_Assert(mask.type() == CV_8UC1);

    int x = cvRound(circle[0]);
    int y = cvRound(circle[1]);
    int radius = cvRound(circle[2]);

    // Create as mask as the intersection of the circle mask and of the ball segmentation mask
    mask = Mat::zeros(segmentation_mask.size(), CV_8U);
    cv::circle(mask, Point(x, y), radius, Scalar(255), FILLED);
    Mat balls_segmentation_mask;
    bitwise_not(segmentation_mask, balls_segmentation_mask); // Negate it as the original mask masks out the balls
    bitwise_and(mask, balls_segmentation_mask, mask);
}

float balls_localizer::get_white_ratio_in_circle_cue(const Mat &src, const Mat &segmentation_mask, Vec3f circle)
{
    CV_Assert(segmentation_mask.type() == CV_8UC1);

    Mat src_hsv;
    cvtColor(src, src_hsv, COLOR_BGR2HSV);

    Mat mask = Mat::zeros(src_hsv.size(), CV_8U);
    get_circle_and_field_mask(segmentation_mask, circle, mask);

    Mat masked_hsv;
    src_hsv.copyTo(masked_hsv, mask);

    Mat white_mask;
    const Vec3b WHITE_HSV_LOWERBOUND = Vec3b(20, 0, 140);
    const Vec3b WHITE_HSV_UPPERBOUND = Vec3b(110, 100, 255);
    inRange(masked_hsv, WHITE_HSV_LOWERBOUND, WHITE_HSV_UPPERBOUND, white_mask);

    // Compute white pixels ratio
    int white_pixels = countNonZero(white_mask);
    int total_circle_pixels = countNonZero(mask);
    double white_ratio = static_cast<double>(white_pixels) / total_circle_pixels;

    return white_ratio;
}

float balls_localizer::get_black_ratio_in_circle(const Mat &src, const Mat &segmentation_mask, Vec3f circle)
{
    CV_Assert(segmentation_mask.type() == CV_8UC1);

    Mat mask = Mat::zeros(src.size(), CV_8U);
    get_circle_and_field_mask(segmentation_mask, circle, mask);

    Mat masked_hsv;
    src.copyTo(masked_hsv, mask);

    Mat black_mask;
    const Vec3b BLACK_HSV_LOWERBOUND = Vec3b(35, 1, 0);
    const Vec3b BLACK_HSV_UPPERBOUND = Vec3b(140, 255, 90);
    inRange(masked_hsv, BLACK_HSV_LOWERBOUND, BLACK_HSV_UPPERBOUND, black_mask);

    // Computing ratio
    int white_pixels = countNonZero(black_mask);
    int total_circle_pixels = countNonZero(mask);
    double percentage_black = static_cast<double>(white_pixels) / total_circle_pixels;

    return percentage_black;
}

void balls_localizer::filter_close_dissimilar_circles(vector<Vec3f> &circles, float neighborhood_distance_threshold, float distance_threshold, float radius_threshold)
{
    vector<bool> to_remove(circles.size(), false);

    for (size_t i = 0; i < circles.size(); ++i)
    {
        for (size_t j = 0; j < circles.size(); ++j)
        {
            // Consider only circles close enough
            if (i != j && norm(circles.at(i) - circles.at(j)) < neighborhood_distance_threshold)
            {
                float y1 = circles.at(i)[1];
                float radius_1 = circles.at(i)[2];
                float y2 = circles.at(j)[1];
                float radius_2 = circles.at(j)[2];

                // Remove circles that are smaller than another circle, below it and enough close to it
                if (y2 > y1 && abs(y2 - y1) < distance_threshold && radius_1 - radius_2 > radius_threshold)
                    to_remove.at(j) = true;
            }
        }
    }

    vector<Vec3f> filtered_circles;
    for (int i = 0; i < circles.size(); i++)
    {
        if (!to_remove[i])
            filtered_circles.push_back(circles.at(i));
    }

    circles = filtered_circles;
}

void balls_localizer::draw_circles(const cv::Mat &src, cv::Mat &dst, vector<cv::Vec3f> &circles)
{
    dst = src.clone();
    for (const Vec3f circle : circles)
    {
        Point center = Point(circle[0], circle[1]);
        cv::circle(dst, center, 1, Scalar(0, 100, 100), 1, LINE_AA);
        int radius = circle[2];
        cv::circle(dst, center, radius, Scalar(255, 0, 255), 1, LINE_AA);
    }
}

void balls_localizer::remove_connected_components_by_diameter(Mat &mask, double min_diameter)
{
    CV_Assert(mask.type() == CV_8UC1);

    Mat labels, stats, centroids;
    int number_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    // Start from 1 to skip the background
    for (int label = 1; label < number_labels; ++label)
    {
        Mat component = (labels == label);

        // Find the minimum enclosing circle
        vector<Point> points;
        findNonZero(component, points);
        Point2f center;
        float radius;
        minEnclosingCircle(points, center, radius);

        double diameter = 2 * radius;
        if (diameter < min_diameter)
            mask.setTo(0, component);
    }
}

float balls_localizer::get_white_ratio_in_circle_stripes(const Mat &src, const Mat &segmentation_mask, Vec3f circle)
{
    CV_Assert(segmentation_mask.type() == CV_8UC1);

    Mat mask = Mat::zeros(src.size(), CV_8U);
    get_circle_and_field_mask(segmentation_mask, circle, mask);

    Mat masked_hsv;
    src.copyTo(masked_hsv, mask);

    Mat white_mask;
    const Vec3b WHITE_HSV_LOWERBOUND = Vec3b(0, 0, 135);
    const Vec3b WHITE_HSV_UPPERBOUND = Vec3b(120, 100, 255);

    // Remove components with small diameter
    const double MIN_DIAMETER = 8;
    inRange(masked_hsv, WHITE_HSV_LOWERBOUND, WHITE_HSV_UPPERBOUND, white_mask);
    remove_connected_components_by_diameter(white_mask, MIN_DIAMETER);

    int white_pixels = countNonZero(white_mask);
    int total_circle_pixels = countNonZero(mask);
    double percentage_white = static_cast<double>(white_pixels) / total_circle_pixels;

    return percentage_white;
}

float balls_localizer::distance_from_middle_hue(const Mat &src, const Mat &segmentation_mask, Vec3f circle)
{
    CV_Assert(segmentation_mask.type() == CV_8UC1);

    Mat src_hsv;
    cvtColor(src, src_hsv, COLOR_BGR2HSV);

    Mat mask = Mat::zeros(src.size(), CV_8U);
    get_circle_and_field_mask(segmentation_mask, circle, mask);

    vector<Mat> channels;
    split(src_hsv, channels);

    // Compute distance of the mean hue value from 128
    const float MIDDLE_HUE = 128;
    return abs(mean(channels[0], mask)[0] - MIDDLE_HUE);
}

void balls_localizer::find_cue_ball(const Mat &src, const Mat &segmentation_mask, const vector<Vec3f> &circles)
{
    // We compute, for each circle, the percentage of "white" pixels. The one with highest percentage is picked as white ball.
    vector<pair<Vec3f, float>> circles_white_ratios;
    for (Vec3f circle : circles)
    {
        circles_white_ratios.push_back({circle, get_white_ratio_in_circle_cue(src, segmentation_mask, circle)});
    }

    //  Sort by descending order of percentage
    sort(circles_white_ratios.begin(), circles_white_ratios.end(), [](const pair<Vec3f, float> &a, const pair<Vec3f, float> &b)
         { return a.second > b.second; });

    Vec3f white_ball_circle;
    float cue_ball_circle_confidence;
    const float MAX_DIFFERENCE_THRESHOLD = 0.1;

    if (circles_white_ratios.at(0).second - circles_white_ratios.at(1).second > MAX_DIFFERENCE_THRESHOLD)
    {
        white_ball_circle = circles_white_ratios.at(0).first;
        cue_ball_circle_confidence = circles_white_ratios.at(0).second;
    }
    else
    {
        /*
            Tie break for very bright balls. In general, the white ball contains shadows pixels similar
            to the board color due to light reflection. For this reason, in general, the hue of the cue
            ball in the shadowed parts will be near the table color hue. So we pick the ball with mean hue nearer
            to the middle hue, that is 128.
        */
        double difference_0 = distance_from_middle_hue(src, segmentation_mask, circles_white_ratios.at(0).first);
        double difference_1 = distance_from_middle_hue(src, segmentation_mask, circles_white_ratios.at(1).first);

        if (difference_0 < difference_1)
        {
            white_ball_circle = circles_white_ratios.at(0).first;
            cue_ball_circle_confidence = circles_white_ratios.at(0).second;
        }
        else
        {
            white_ball_circle = circles_white_ratios.at(1).first;
            cue_ball_circle_confidence = circles_white_ratios.at(1).second;
        }
    }

    localization.cue.circle = white_ball_circle;
    localization.cue.bounding_box = get_bounding_box(white_ball_circle);
    localization.cue.confidence = cue_ball_circle_confidence;
}

void balls_localizer::find_black_ball(const Mat &src, const Mat &segmentation_mask, const vector<Vec3f> &circles)
{
    Mat src_hsv;
    cvtColor(src, src_hsv, COLOR_BGR2HSV);

    // We compute, for each circle, the percentage of "white" pixels. The one with highest percentage is picked as white ball.
    vector<pair<Vec3f, float>> circles_black_ratios;
    for (Vec3f circle : circles)
    {
        circles_black_ratios.push_back({circle, get_black_ratio_in_circle(src_hsv, segmentation_mask, circle)});
    }

    // Sort by descending order of percentage
    sort(circles_black_ratios.begin(), circles_black_ratios.end(), [](const pair<Vec3f, float> &a, const pair<Vec3f, float> &b)
         { return a.second > b.second; });

    localization.black.circle = circles_black_ratios.at(0).first;
    localization.black.bounding_box = get_bounding_box(circles_black_ratios.at(0).first);
    localization.black.confidence = circles_black_ratios.at(0).second;
}

void balls_localizer::find_stripe_balls(const Mat &src, const Mat &segmentation_mask, const vector<Vec3f> &circles)
{
    Mat src_hsv;
    cvtColor(src, src_hsv, COLOR_BGR2HSV);

    // We compute, for each circle, the percentage of "white" pixels.
    vector<pair<Vec3f, float>> circles_white_ratios;
    for (Vec3f circle : circles)
    {
        circles_white_ratios.push_back({circle, get_white_ratio_in_circle_stripes(src_hsv, segmentation_mask, circle)});
    }

    vector<pair<Vec3f, float>> circles_white_ratios_filtered;
    copy_if(circles_white_ratios.begin(), circles_white_ratios.end(), back_inserter(circles_white_ratios_filtered), [](pair<Vec3f, float> p)
            {  
                    const float LOW_THRESHOLD = 0.15;
                    const float HIGH_THRESHOLD = 0.81;
                    return p.second >= LOW_THRESHOLD && p.second <= HIGH_THRESHOLD; });

    vector<pair<Vec3f, float>> stripes_circles;
    // Exclude white and black balls, since they may be incorrectly be detected as stripes
    copy_if(circles_white_ratios_filtered.begin(), circles_white_ratios_filtered.end(), back_inserter(stripes_circles), [this](pair<Vec3f, float> p)
            { return p.first != this->localization.cue.circle && p.first != this->localization.black.circle; });

    localization.stripes.clear();
    for (const pair<Vec3f, float> &pair : stripes_circles)
    {
        ball_localization stripe_localization;
        stripe_localization.circle = pair.first;
        stripe_localization.bounding_box = get_bounding_box(pair.first);
        stripe_localization.confidence = pair.second;
        localization.stripes.push_back(stripe_localization);
    }
}

void balls_localizer::find_solid_balls(const Mat &src, const Mat &segmentation_mask, const vector<Vec3f> &circles)
{
    vector<Vec3f> solids_circles;
    // Exclude cue, black and stripes
    copy_if(circles.begin(), circles.end(), back_inserter(solids_circles), [this](Vec3f circle)
            {
        if (circle == this->localization.cue.circle)
            return false;
        if (circle == this->localization.black.circle)
            return false;
        for (ball_localization stripe_localization : this->localization.stripes)
        {
            if (circle == stripe_localization.circle)
                return false;
        }
        return true; });

    /*
        Since the classification of a ball being solid depends exclusively on the classification
        of the other kinds of balls, the confidence of a ball being solid can be estimated as the
        average of the other confidences.
    */
    float solid_confidence = 0;
    solid_confidence += localization.cue.confidence;
    solid_confidence += localization.black.confidence;
    for (const ball_localization &stripe_localization : localization.stripes)
    {
        solid_confidence += stripe_localization.confidence;
    }
    solid_confidence /= (localization.stripes.size() + 2); // +2 represents cue and black

    // All the other must be solids
    for (Vec3f circle : solids_circles)
    {
        ball_localization solid_localization;
        solid_localization.circle = circle;
        solid_localization.bounding_box = get_bounding_box(circle);
        solid_localization.confidence = solid_confidence;
        localization.solids.push_back(solid_localization);
    }
}