#include <iostream>
#include <cmath>
#include <map>
#include <queue>
#include <opencv2/features2d.hpp>
#include "balls_localizer.h"
#include "utils.h"

using namespace cv;
using namespace std;

void balls_localizer::localize(const Mat &src, const Mat &mask, const vector<Point> playing_field_corners)
{
    const int FILTER_SIZE = 3;
    const int FILTER_SIGMA = 5;
    Mat blurred;
    dilate(src, blurred, getStructuringElement(MORPH_CROSS, Size(3, 3)));
    //GaussianBlur(blurred.clone(), blurred, Size(FILTER_SIZE, FILTER_SIZE), FILTER_SIGMA, FILTER_SIGMA);

    Mat masked = blurred.clone();
    Mat mask_bgr;
    cvtColor(mask, mask_bgr, COLOR_GRAY2BGR);
    bitwise_and(masked, mask_bgr, masked);

    //imshow("", masked);
    //waitKey();

    Mat connected_components_segmentation;
    segmentation(masked, connected_components_segmentation);
    //imshow("", connected_components_segmentation);
    //waitKey(0);

    /*
    const int RADIUS = 30;
    Vec3b board_color = get_board_color(connected_components_segmentation, RADIUS);
    Mat connected_components_segmentation_mask;
    inRange(connected_components_segmentation, board_color, board_color, connected_components_segmentation_mask);
    // segmented.setTo(Scalar(0, 0, 0), connected_components_segmentation_mask);
    imshow("", connected_components_segmentation_mask);
    waitKey();
    
    // Single opening
    // morphologyEx(connected_components_segmentation_mask.clone(), connected_components_segmentation_mask, MORPH_OPEN, getStructuringElement(MORPH_CROSS, Size(3, 3)));
    // imshow("", connected_components_segmentation_mask);
    // waitKey();

    Mat connected_components_pixels = src.clone();
    vector<Point> seed_points;

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(connected_components_segmentation_mask, labels, stats, centroids);

    // Iterate through each connected component (excluding the background)
    for (int i = 1; i < nLabels; ++i)
    {
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        ;

        // Calculate the center of the bounding box
        int centerX = left + width / 2;
        int centerY = top + height / 2;

        // Ensure the point is within the component
        cv::Point innerPoint(centerX, centerY);

        // Check if the point is within the component (if necessary)
        if (labels.at<int>(centerY, centerX) != i)
        {
            // Adjust the point to be within the component
            for (int y = top; y < top + height; ++y)
            {
                for (int x = left; x < left + width; ++x)
                {
                    if (labels.at<int>(y, x) == i)
                    {
                        innerPoint = cv::Point(x, y);
                        break;
                    }
                }
                if (labels.at<int>(innerPoint.y, innerPoint.x) == i)
                    break;
            }
        }

        cv::circle(connected_components_pixels, innerPoint, 3, cv::Scalar(255, 0, 0), -1);
        seed_points.push_back(innerPoint);
    }

    imshow("", connected_components_pixels);
    waitKey();

    Mat segmentation_mask;
    region_growing(masked, segmentation_mask, seed_points, 5, 5, 5);

    imshow("", segmentation_mask);
    waitKey(0);

    vector<Vec3f> circles;

    HoughCircles(segmentation_mask, circles, HOUGH_GRADIENT_ALT, 5, 10, 100, 0.2, 5, 21);
    // HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, 18, 30, 1, 5, 17);

    Mat display = src.clone();
    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        circle(display, center, 1, Scalar(0, 100, 100), 1, LINE_AA);
        int radius = c[2];
        circle(display, center, radius, Scalar(255, 0, 255), 1, LINE_AA);
    }
    imshow("", display);
    waitKey(0);

    vector<Mat> hough_masks;
    circles_masks(circles, hough_masks, src.size());

    vector<Vec3f> filtered_circles;
    vector<Mat> filtered_masks;
    filter_empty_circles(circles, hough_masks, segmentation_mask, filtered_circles, filtered_masks, 0.60);

    vector<Vec3f> filtered_out_of_bounds_circles;
    filter_out_of_bound_circles(filtered_circles, mask, filtered_out_of_bounds_circles, 20);

    display = src.clone();
    for (size_t i = 0; i < filtered_out_of_bounds_circles.size(); i++)
    {
        Vec3i c = filtered_out_of_bounds_circles[i];
        Point center = Point(c[0], c[1]);
        circle(display, center, 1, Scalar(0, 100, 100), 1, LINE_AA);
        int radius = c[2];
        circle(display, center, radius, Scalar(255, 0, 255), 1, LINE_AA);
    }
    imshow("", display);
    waitKey(0);

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
    */
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

void balls_localizer::filter_out_of_bound_circles_perspective(const std::vector<cv::Vec3f> &circles, const Mat &table_mask, std::vector<cv::Vec3f> &filtered_circles, int distance_threshold)
{
    Mat shrinked_table_mask;
    erode(table_mask, shrinked_table_mask, getStructuringElement(MORPH_RECT, Size(distance_threshold, 1)));

    for (Vec3f circle : circles)
    {
        Point center = Point(circle[0], circle[1]);
        if (shrinked_table_mask.at<uchar>(center) == 255)
        {
            filtered_circles.push_back(circle);
        }
    }
}

void balls_localizer::segmentation(const Mat &src, Mat &dst)
{
    imshow("", src);
    waitKey(0);

    cvtColor(src, dst, COLOR_BGR2HSV);

    const int VALUE_UNIFORM = 255;
    vector<Mat> hsv_channels;
    split(dst, hsv_channels);
    hsv_channels[1].setTo(VALUE_UNIFORM);
    hsv_channels[2].setTo(VALUE_UNIFORM);
    merge(hsv_channels, dst);

    imshow("", dst);
    waitKey(0);

    Vec3b color = get_board_color(dst, 30);
    cout << color << endl;
    
    //Mat mask = Mat::zeros(dst.rows, dst.cols, CV_8U);
    Mat mask(dst.rows, dst.cols, CV_8UC1, Scalar(0));
    for(int y=0;y<dst.rows;y++)
    {
        for(int x=0;x<dst.cols;x++)
        {
            if(abs(dst.at<Vec3b>(y,x)[0]-color[0]) > 5)
                mask.at<uchar>(y,x) = 255;
            
        }
    }
    
    imshow("", mask);
    waitKey(0);
    dilate(mask.clone(), mask, getStructuringElement(MORPH_RECT, Size(5, 5)));
    imshow("", mask);
    waitKey(0);
    vector<Vec3f> circles;
    //HoughCircles(mask, circles, HOUGH_GRADIENT_ALT, 5, 10, 100, 0.2, 5, 21);
    //HoughCircles(mask, circles, HOUGH_GRADIENT_ALT, 5, 10, 100, 0.1, 5, 40);

    //HoughCircles(mask, circles, HOUGH_GRADIENT_ALT, 5, 10, 150, 0.2, 5, 60);
    HoughCircles(mask, circles, HOUGH_GRADIENT_ALT, 5, 10, 200, 0.08, 5, 60);

    // HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, 18, 30, 1, 5, 17);

    Mat d = src.clone();
    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        circle(d, center, 1, Scalar(0, 100, 100), 1, LINE_AA);
        int radius = c[2];
        circle(d, center, radius, Scalar(255, 0, 255), 1, LINE_AA);
    }
    imshow("DISPLAY", d);
    waitKey(0);


    /*
    // data contains dst data (init with src data) used for kmeans clustering (therefore employs 32-bit float values)
    Mat data;
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // Image segmentation is performed via kmeans on the hsv img
    Mat labels, centers;
    // const int NUMBER_OF_CENTERS = 8;
    const int NUMBER_OF_CENTERS = 10;
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
    dst.convertTo(dst, CV_8U);*/
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

    int dx[] = {-1, 1, 0, 0, 1, 1, -1, -1};
    int dy[] = {0, 0, -1, 1, 1, -1, 1, -1};

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
