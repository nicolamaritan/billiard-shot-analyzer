#include "balls_localizer.h"
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

void extract_seed_points(const Mat &inrange_segmentation_mask, vector<Point> &seed_points)
{
    // Ensure the seed_points vector is empty
    seed_points.clear();

    // Loop through each pixel in the mask
    for (int y = 0; y < inrange_segmentation_mask.rows; ++y)
    {
        for (int x = 0; x < inrange_segmentation_mask.cols; ++x)
        {
            // Check if the pixel value is non-zero
            if (inrange_segmentation_mask.at<uchar>(y, x) > 0)
            {
                // Add the point to the vector
                seed_points.push_back(Point(x, y));
            }
        }
    }
}

void fill_small_holes(cv::Mat &binary_mask, double area_threshold)
{
    CV_Assert(binary_mask.type() == CV_8UC1);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); ++i)
    {
        double area = cv::contourArea(contours[i]);

        // If the area is less than the threshold, fill the hole
        if (area < area_threshold)
        {
            cv::drawContours(binary_mask, contours, i, cv::Scalar(255), cv::FILLED, 8, hierarchy, 1);
        }
    }
}

void balls_localizer::localize(const Mat &src)
{
    const int FILTER_SIZE = 3;
    const int FILTER_SIGMA = 3;
    Mat blurred;
    GaussianBlur(src, blurred, Size(FILTER_SIZE, FILTER_SIZE), FILTER_SIGMA, FILTER_SIGMA);

    Mat masked = blurred.clone();
    Mat mask_bgr;
    cvtColor(playing_field_mask, mask_bgr, COLOR_GRAY2BGR);
    bitwise_and(masked, mask_bgr, masked);

    vector<Point> seed_points;
    Mat inrange_segmentation_mask_board, inrange_segmentation_mask_shadows, segmentation_mask;
    Mat masked_hsv;
    cvtColor(masked, masked_hsv, COLOR_BGR2HSV);

    Vec3b board_color_hsv = get_board_color(masked_hsv, 100);
    Vec3b shadow_hsv = board_color_hsv - Vec3b(0, 0, 90);
    inRange(masked_hsv, board_color_hsv - Vec3b(4, 50, 40), board_color_hsv + Vec3b(4, 40, 15), inrange_segmentation_mask_board);
    inRange(masked_hsv, shadow_hsv - Vec3b(7, 100, 100), shadow_hsv + Vec3b(7, 80, 80), inrange_segmentation_mask_shadows);
    
    Mat inrange_segmentation_mask_black, inrange_segmentation_mask_white; 
    inRange(masked, Vec3b(0, 0, 0), Vec3b(40, 40, 40), inrange_segmentation_mask_black);
    inRange(masked, Vec3b(200, 200, 200), Vec3b(255, 255, 255), inrange_segmentation_mask_white);
    morphologyEx(inrange_segmentation_mask_shadows.clone(), inrange_segmentation_mask_shadows, MORPH_OPEN, getStructuringElement(MORPH_CROSS, Size(3, 3)));

    Mat outer_field;
    Mat shrinked_playing_field_mask;
    erode(playing_field_mask, shrinked_playing_field_mask, getStructuringElement(MORPH_CROSS, Size(70, 70)));
    bitwise_not(shrinked_playing_field_mask, outer_field);
    imshow("outer_field", outer_field);

    bitwise_and(inrange_segmentation_mask_shadows.clone(), outer_field, inrange_segmentation_mask_shadows);

    imshow("inrange_sementation_1", inrange_segmentation_mask_board);
    imshow("inrange_sementation_2", inrange_segmentation_mask_shadows);
    bitwise_or(inrange_segmentation_mask_board, inrange_segmentation_mask_shadows, segmentation_mask);

    extract_seed_points(segmentation_mask, seed_points);
    region_growing(masked_hsv, segmentation_mask, seed_points, 3, 6, 4);

    vector<Point> mask_seed_points;
    Mat out_of_field_mask;
    mask_seed_points.push_back(Point(0, 0));
    mask_region_growing(segmentation_mask, out_of_field_mask, mask_seed_points);

    bitwise_or(segmentation_mask.clone(), out_of_field_mask, segmentation_mask);
    fill_small_holes(segmentation_mask, 90);
    imshow("segmentation", segmentation_mask);

    Mat display_segm, inrange_segmentation_mask_bgr;
    cvtColor(inrange_segmentation_mask_board, inrange_segmentation_mask_bgr, COLOR_GRAY2BGR);
    bitwise_and(masked, inrange_segmentation_mask_bgr, display_segm);

    vector<Vec3f> circles;
    HoughCircles(segmentation_mask, circles, HOUGH_GRADIENT, 0.3, 18, 100, 5, 8, 16);

    Mat display = src.clone();
    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        circle(display, center, 1, Scalar(0, 100, 100), 1, LINE_AA);
        int radius = c[2];
        circle(display, center, radius, Scalar(255, 0, 255), 1, LINE_AA);
    }
    // imshow("", display);
    // waitKey(0);

    vector<Mat> hough_masks;
    circles_masks(circles, hough_masks, src.size());

    vector<Vec3f> filtered_circles;
    vector<Mat> filtered_masks;
    filter_empty_circles(circles, hough_masks, segmentation_mask, filtered_circles, filtered_masks, 0.60);

    vector<Vec3f> filtered_out_of_bounds_circles;
    filter_out_of_bound_circles(filtered_circles, playing_field_mask, filtered_out_of_bounds_circles, 20);

    vector<Vec3f> filtered_near_holes_circles;
    assert(playing_field_hole_points.size() != 0);
    filter_near_holes_circles(filtered_out_of_bounds_circles, filtered_near_holes_circles, playing_field_hole_points, 27);

    display = blurred.clone();
    for (size_t i = 0; i < filtered_near_holes_circles.size(); i++)
    {
        Vec3i c = filtered_near_holes_circles[i];
        Point center = Point(c[0], c[1]);
        circle(display, center, 1, Scalar(0, 100, 100), 2, LINE_AA);
        int radius = c[2];
        circle(display, center, radius, Scalar(255, 0, 255), 2, LINE_AA);
    }
    imshow("", display);
    // waitKey(0);

    for (size_t i = 0; i < filtered_circles.size(); i++)
    {
        Mat single_circle_filtered_mask;
        Mat blurred_masked;
        cvtColor(filtered_masks[i], single_circle_filtered_mask, COLOR_GRAY2BGR);
        bitwise_and(blurred, single_circle_filtered_mask, blurred_masked);
        Mat all_hsv;
        cvtColor(blurred_masked, all_hsv, COLOR_BGR2HSV);

        vector<Mat> hsv_channels;
        split(all_hsv, hsv_channels);
    }
}

void balls_localizer::circles_masks(const std::vector<Vec3f> &circles, std::vector<Mat> &masks, Size mask_size)
{
    for (size_t i = 0; i < circles.size(); i++)
    {
        Mat mask(mask_size, CV_8U);
        mask.setTo(Scalar(0));
        circle(mask, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(255), FILLED);
        masks.push_back(mask);
    }
}

void balls_localizer::filter_empty_circles(const std::vector<cv::Vec3f> &circles, const std::vector<Mat> &masks, const Mat &segmentation_mask, std::vector<cv::Vec3f> &filtered_circles, std::vector<cv::Mat> &filtered_masks, float intersection_threshold)
{
    for (size_t i = 0; i < circles.size(); i++)
    {
        float circle_area = countNonZero(masks[i]);
        Mat masks_intersection;
        bitwise_and(masks[i], segmentation_mask, masks_intersection);
        float intersection_area = countNonZero(masks_intersection);

        if (intersection_area / circle_area < intersection_threshold)
        {
            filtered_circles.push_back(circles[i]);
            filtered_masks.push_back(masks[i]);
        }
    }
}

void balls_localizer::filter_out_of_bound_circles(const std::vector<cv::Vec3f> &circles, const Mat &table_mask, std::vector<cv::Vec3f> &filtered_circles, int distance_threshold)
{
    Mat shrinked_table_mask;
    erode(table_mask, shrinked_table_mask, getStructuringElement(MORPH_CROSS, Size(distance_threshold, distance_threshold)));

    for (Vec3f circle : circles)
    {
        Point center = Point(circle[0], circle[1]);
        if (shrinked_table_mask.at<uchar>(center) == 255)
        {
            filtered_circles.push_back(circle);
        }
    }
}

void balls_localizer::filter_near_holes_circles(const std::vector<cv::Vec3f> &circles, std::vector<cv::Vec3f> &filtered_circles, const vector<Point> &holes_points, float distance_threshold)
{
    for (Vec3f circle : circles)
    {
        Point circle_point = Point(static_cast<int>(circle[0]), static_cast<int>(circle[1]));
        bool keep_circle = true;
        for (Point hole_point : holes_points)
        {
            if (norm(hole_point - circle_point) < distance_threshold)
                keep_circle = false;
        }

        if (keep_circle)
            filtered_circles.push_back(circle);
    }
}

void balls_localizer::segmentation(const Mat &src, Mat &dst)
{
    // HSV allows to separate brightness from other color characteristics, therefore
    // it is employed for kmeans clustering.
    cvtColor(src, dst, COLOR_BGR2HSV);

    const int VALUE_UNIFORM = 255;
    vector<Mat> hsv_channels;
    split(dst, hsv_channels);
    hsv_channels[1].setTo(VALUE_UNIFORM);
    hsv_channels[2].setTo(VALUE_UNIFORM);
    merge(hsv_channels, dst);

    const int NUMBER_OF_CENTERS = 8;
    kmeans(src, dst, NUMBER_OF_CENTERS);
}

Vec3b balls_localizer::get_board_color(const Mat &src, float radius)
{
    int center_cols = src.cols / 2;
    int center_rows = src.rows / 2;
    vector<Vec3b> pixel_values;

    // Collect all pixel values in a radius 'radius' around the image center.
    for (int row = -radius; row <= radius; ++row)
    {
        for (int col = -radius; col <= radius; ++col)
        {
            if (col * col + row * row <= radius * radius)
            {
                int current_row = center_rows + row;
                int current_col = center_cols + col;

                if (current_row >= 0 && current_row < src.rows && current_col >= 0 && current_col < src.cols)
                {
                    pixel_values.push_back(src.at<Vec3b>(current_row, current_col));
                }
            }
        }
    }

    // Return black if no pixel_values are collected
    if (pixel_values.empty())
    {
        return Vec3b(0, 0, 0);
    }

    // Sort by norm. In a grayscale context, we would have just considered the pixel intensity.
    // However, now we have 3 components. So we sort the pixel values triplets by their norm.
    sort(pixel_values.begin(), pixel_values.end(), [](const Vec3b &a, const Vec3b &b)
         { return norm(a) < norm(b); });

    return pixel_values[pixel_values.size() / 2];
}
