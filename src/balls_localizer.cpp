#include <iostream>
#include <cmath>
#include <map>
#include <queue>
#include <opencv2/features2d.hpp>
#include "balls_localizer.h"

using namespace cv;
using namespace std;

void balls_localizer::localize(const Mat &src, const Mat &mask)
{
    const int FILTER_SIZE = 3;
    const int FILTER_SIGMA = 20;
    Mat blurred;
    GaussianBlur(src.clone(), blurred, Size(FILTER_SIZE, FILTER_SIZE), FILTER_SIGMA, FILTER_SIGMA);

    Mat masked = blurred.clone();
    Mat mask_bgr;
    cvtColor(mask, mask_bgr, COLOR_GRAY2BGR);
    bitwise_and(masked, mask_bgr, masked);

    imshow("", masked);
    waitKey();

    Mat connected_components_segmentation, labels;
    segmentation(masked, connected_components_segmentation);
    
    
    imshow("", connected_components_segmentation);
    waitKey(0);

    const int RADIUS = 30;
    Vec3b board_color = get_board_color(connected_components_segmentation, RADIUS);
    Mat connected_components_segmentation_mask;
    inRange(connected_components_segmentation, board_color, board_color, connected_components_segmentation_mask);
    // segmented.setTo(Scalar(0, 0, 0), connected_components_segmentation_mask);
    imshow("", connected_components_segmentation_mask);
    waitKey();

    // Single opening
    morphologyEx(connected_components_segmentation_mask.clone(), connected_components_segmentation_mask, MORPH_OPEN, getStructuringElement(MORPH_CROSS, Size(3, 3)));
    imshow("", connected_components_segmentation_mask);
    waitKey();

    Mat connected_components_pixels = src.clone();
    vector<Point> seed_points;
    // Find contours (connected components)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(connected_components_segmentation_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Iterate through each contour
    for (size_t i = 0; i < contours.size(); ++i)
    {
        // Get an arbitrary pixel from the contour (first point in the contour)
        cv::Point arbitraryPixel = contours[i][contours[i].size() - 1];
        seed_points.push_back(arbitraryPixel);
        //std::cout << "Component " << i + 1 << ": " << arbitraryPixel << std::endl;

        // Optional: Draw the pixel on the image for visualization
        cv::circle(connected_components_pixels, arbitraryPixel, 3, cv::Scalar(255, 0, 0), -1);
    }
    imshow("", connected_components_pixels);
    waitKey();

    Mat segmentation_mask;
    region_growing(masked, segmentation_mask, seed_points, 3, 10, 255);

    imshow("", segmentation_mask);
    waitKey(0);

    //non_maxima_connected_component_suppression(segmentation_mask.clone(), segmentation_mask);
    //imshow("", segmentation_mask);
    //waitKey();
    vector<Vec3f> circles;

    HoughCircles(segmentation_mask, circles, HOUGH_GRADIENT_ALT, 2, 10, 100, 0.2, 5, 20);
    // HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, 18, 30, 1, 5, 17);

    Mat display = src.clone();
    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(display, center, 1, Scalar(0, 100, 100), 1, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(display, center, radius, Scalar(255, 0, 255), 1, LINE_AA);
    }

    // Display the result
    imshow("", display);
    waitKey(0);
}

void balls_localizer::segmentation(const Mat &src, Mat &dst)
{
    // HSV allows to separate brightness from other color characteristics, therefore
    // it is employed for kmeans clustering.
    cvtColor(src, dst, COLOR_BGR2HSV);

    imshow("", dst);
    waitKey();

    // data contains dst data (init with src data) used for kmeans clustering (therefore employs 32-bit float values)
    Mat data;
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // Image segmentation is performed via kmeans on the hsv img
    Mat labels, centers;
    const int NUMBER_OF_CENTERS = 8;
    const int KMEANS_MAX_COUNT = 10;
    const int KMEANS_EPSILON = 1.0;
    const int KMEANS_ATTEMPTS = 3;
    kmeans(data, NUMBER_OF_CENTERS, labels, TermCriteria(TermCriteria::MAX_ITER, KMEANS_MAX_COUNT, KMEANS_EPSILON), KMEANS_ATTEMPTS, KMEANS_PP_CENTERS, centers);

    // Reshape both to a single row of Vec3f pixels
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    // Replace pixel values with their centroids value
    for (int i = 0; i < data.rows; i++)
    {
        int center_id = labels.at<int>(i);
        data.at<Vec3f>(i) = centers.at<Vec3f>(center_id);
    }

    dst = data.reshape(3, dst.rows);
    dst.convertTo(dst, CV_8U);
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

void balls_localizer::non_maxima_connected_component_suppression(const Mat &src, Mat &dst)
{
    src.copyTo(dst);
    Mat connected_components_labels, stats, centroids;
    connectedComponentsWithStats(src, connected_components_labels, stats, centroids);

    const int AREA_STAT_ID = 4;
    int max_label_component = 1;
    int max_area = 0;

    // Find component with greatest area
    for (int i = 1; i < stats.rows; i++)
    {
        int component_area = stats.at<int>(i, AREA_STAT_ID);
        if (component_area > max_area)
        {
            max_area = component_area;
            max_label_component = i;
        }
    }

    // Suppress (mask set to 0) all components with non greatest area
    for (int row = 0; row < src.rows; row++)
    {
        for (int col = 0; col < src.cols; col++)
        {
            if (connected_components_labels.at<int>(row, col) != max_label_component)
            {
                dst.at<uchar>(row, col) = 0;
            }
        }
    }
}

void balls_localizer::region_growing(const Mat &src, Mat &dst, const vector<Point> &seeds, int threshold_0, int threshold_1, int threshold_2)
{
    dst = Mat::zeros(src.size(), CV_8UC1); // Initialize the destination image
    queue<Point> toGrow;                   // Queue for points to be processed

    for (const Point &seed : seeds)
    {
        if (seed.x >= 0 && seed.x < src.cols && seed.y >= 0 && seed.y < src.rows)
        {
            toGrow.push(seed);
            dst.at<uchar>(seed) = 255; // Mark the seed point in the destination image
        }
    }

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    while (!toGrow.empty())
    {
        Point p = toGrow.front();
        toGrow.pop();

        for (int i = 0; i < 4; ++i)
        {
            Point neighbor(p.x + dx[i], p.y + dy[i]);
            if (neighbor.x >= 0 && neighbor.x < src.cols && neighbor.y >= 0 && neighbor.y < src.rows)
            {
                if (dst.at<uchar>(neighbor) == 0 &&
                    abs(src.at<Vec3b>(p)[0] - src.at<Vec3b>(neighbor)[0]) <= threshold_0 &&
                    abs(src.at<Vec3b>(p)[1] - src.at<Vec3b>(neighbor)[1]) <= threshold_1 &&
                    abs(src.at<Vec3b>(p)[2] - src.at<Vec3b>(neighbor)[2]) <= threshold_2)
                {
                    dst.at<uchar>(neighbor) = 255;
                    toGrow.push(neighbor);
                }
            }
        }
    }
}
