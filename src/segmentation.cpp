// Author: Nicola Maritan 2121717

#include "segmentation.h"

#include <queue>

using namespace std;
using namespace cv;

void kmeans(const Mat &src, Mat &dst, int centroids)
{
    if (src.empty())
    {
        const string INVALID_EMPTY_MAT = "Invalid empty mat for kmeans.";
        throw invalid_argument(INVALID_EMPTY_MAT);
    }

    // data contains dst data (init with src data) used for kmeans clustering (therefore employs 32-bit float values).
    Mat data;
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // Image segmentation is performed via kmeans on the hsv img.
    Mat labels, centers;
    const int KMEANS_MAX_COUNT = 10;
    const int KMEANS_EPSILON = 1.0;
    const int KMEANS_ATTEMPTS = 8;
    kmeans(data, centroids, labels, TermCriteria(TermCriteria::MAX_ITER, KMEANS_MAX_COUNT, KMEANS_EPSILON), KMEANS_ATTEMPTS, KMEANS_PP_CENTERS, centers);

    // Reshape both to a single row of Vec3f pixels.
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    // Replace pixel values with their centroids value.
    for (int i = 0; i < data.rows; i++)
    {
        int center_id = labels.at<int>(i);
        data.at<Vec3f>(i) = centers.at<Vec3f>(center_id);
    }

    dst = data.reshape(3, dst.rows);
    dst.convertTo(dst, CV_8U);
}

void region_growing(const Mat &src, Mat &dst, const vector<Point> &seeds, int threshold_0, int threshold_1, int threshold_2)
{
    if (src.empty())
    {
        const string INVALID_EMPTY_MAT = "Invalid empty mat for region growing.";
        throw invalid_argument(INVALID_EMPTY_MAT);
    }

    dst = Mat::zeros(src.size(), CV_8UC1); // Initialize the destination image.
    queue<Point> to_grow; // Queue for points to be processed.

    for (const Point &seed : seeds)
    {
        if (seed.x >= 0 && seed.x < src.cols && seed.y >= 0 && seed.y < src.rows)
        {
            to_grow.push(seed);
            dst.at<uchar>(seed) = 255; // Mark the seed point in the destination image.
        }
    }
    const pair<int, int> LEFT = {-1, 0}; // Left direction.
    const pair<int, int> RIGHT = {1, 0}; // Right direction.
    const pair<int, int> DOWN = {0, -1}; // Down direction.
    const pair<int, int> UP = {0, 1}; // Up direction.

    const vector<pair<int, int>> directions = {LEFT, RIGHT, DOWN, UP}; // directions of expantions of growing process.

    // Growing process.
    while (!to_grow.empty())
    {
        Point p = to_grow.front();
        to_grow.pop();

        for (pair<int, int> direction : directions)
        {
            Point neighbor(p.x + direction.first, p.y + direction.second);
            if (neighbor.x >= 0 && neighbor.x < src.cols && neighbor.y >= 0 && neighbor.y < src.rows)
            {
                if (dst.at<uchar>(neighbor) == 0 &&
                    abs(src.at<Vec3b>(p)[0] - src.at<Vec3b>(neighbor)[0]) <= threshold_0 &&
                    abs(src.at<Vec3b>(p)[1] - src.at<Vec3b>(neighbor)[1]) <= threshold_1 &&
                    abs(src.at<Vec3b>(p)[2] - src.at<Vec3b>(neighbor)[2]) <= threshold_2)
                {
                    dst.at<uchar>(neighbor) = 255;
                    to_grow.push(neighbor);
                }
            }
        }
    }
}

void mask_region_growing(const Mat &src, Mat &dst, const vector<Point> &seeds)
{
    Mat src_bgr;
    cvtColor(src, src_bgr, COLOR_GRAY2BGR);
    region_growing(src_bgr, dst, seeds, 0, 0, 0);
}

Vec3b get_playing_field_color(const Mat &src, float radius)
{
    if (src.empty())
    {
        const string INVALID_EMPTY_MAT = "Invalid empty mat for getting playing field color.";
        throw invalid_argument(INVALID_EMPTY_MAT);
    }

    int center_cols = src.cols / 2;
    int center_rows = src.rows / 2;
    vector<Vec3b> pixel_values;

    // Collect all pixel values in a radius of value $radius around the image center.
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

    // Return black if no pixel_values are collected.
    if (pixel_values.empty())
        return Vec3b(0, 0, 0);

    /*
        Sort by norm. In a grayscale context, we would have just considered the pixel intensity.
        However, now we have 3 components. So we sort the pixel values triplets by their norm.
    */
    sort(pixel_values.begin(), pixel_values.end(), [](const Vec3b &a, const Vec3b &b)
         { return norm(a) < norm(b); });

    return pixel_values[pixel_values.size() / 2];
}