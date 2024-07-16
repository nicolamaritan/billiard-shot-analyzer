#include "segmentation.h"
#include <queue>

using namespace std;
using namespace cv;

void kmeans(const Mat &src, Mat &dst, int centroids)
{
    // data contains dst data (init with src data) used for kmeans clustering (therefore employs 32-bit float values)
    Mat data;
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // Image segmentation is performed via kmeans on the hsv img
    Mat labels, centers;
    const int KMEANS_MAX_COUNT = 10;
    const int KMEANS_EPSILON = 1.0;
    const int KMEANS_ATTEMPTS = 3;
    kmeans(data, centroids, labels, TermCriteria(TermCriteria::MAX_ITER, KMEANS_MAX_COUNT, KMEANS_EPSILON), KMEANS_ATTEMPTS, KMEANS_PP_CENTERS, centers);

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

void region_growing(const Mat &src, Mat &dst, const vector<Point> &seeds, int threshold_0, int threshold_1, int threshold_2)
{
    dst = Mat::zeros(src.size(), CV_8UC1); // Initialize the destination image
    queue<Point> to_grow;                   // Queue for points to be processed

    for (const Point &seed : seeds)
    {
        if (seed.x >= 0 && seed.x < src.cols && seed.y >= 0 && seed.y < src.rows)
        {
            to_grow.push(seed);
            dst.at<uchar>(seed) = 255; // Mark the seed point in the destination image
        }
    }

    vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

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
