#include <iostream>
#include <cmath>
#include <map>
#include "playing_field_localizer.h"

using namespace cv;
using namespace std;

void playing_field_localizer::segmentation(const Mat &src, Mat &dst)
{
    Mat src_hsv;
    cvtColor(src, src_hsv, COLOR_BGR2HSV);
    dst = src_hsv;

    Mat data;
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // do kmeans
    Mat labels, centers;
    kmeans(data, 4, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 1.0), 3,
           KMEANS_PP_CENTERS, centers);

    // reshape both to a single row of Vec3f pixels:
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    // replace pixel values with their center value:
    Vec3f *p = data.ptr<Vec3f>();
    for (size_t i = 0; i < data.rows; i++)
    {
        int center_id = labels.at<int>(i);

        if (center_id == 0)
        {
            p[i] = centers.at<Vec3f>(center_id);
        }
        else
        {
            p[i] = Vec3f(0, 0, 0);
        }
        p[i] = centers.at<Vec3f>(center_id);
    }

    dst = data.reshape(3, dst.rows);
    dst.convertTo(dst, CV_8U);
}

cv::Vec3b playing_field_localizer::get_board_color(const cv::Mat &src)
{
    return src.at<Vec3b>(src.rows / 2, src.cols / 2);
}

void playing_field_localizer::find_lines(const cv::Mat &edges)
{
    Mat cdst;

    // Copy edges to the images that will display the results in BGR
    cvtColor(edges, cdst, COLOR_GRAY2BGR);
    // Standard Hough Line Transform
    vector<Vec2f> lines;                                 // will hold the results of the detection
    HoughLines(edges, lines, 1.6, 1.8 * CV_PI / 180, 120, 0, 0); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    imshow("", cdst);
    waitKey();
}

void playing_field_localizer::localize(const Mat &src, Mat &dst)
{
    GaussianBlur(src.clone(), src, Size(3, 3), 12, 12);

    Mat segmented, labels;
    segmentation(src, segmented);

    imshow("", segmented);
    waitKey(0);

    Vec3b board_color = get_board_color(segmented);

    Mat mask;
    inRange(segmented, board_color, board_color, mask);
    segmented.setTo(Scalar(0, 0, 0), mask);

    imshow("", mask);
    waitKey(0);

    Mat element = getStructuringElement(MORPH_CROSS, Size(5, 5));
    morphologyEx(mask.clone(), mask, MORPH_OPEN, element);
    imshow("", mask);
    waitKey(0);

    element = getStructuringElement(MORPH_RECT, Size(20, 20));
    morphologyEx(mask.clone(), mask, MORPH_CLOSE, element);
    imshow("", mask);
    waitKey(0);

    Mat edges;
    Canny(mask, edges, 50, 150);
    imshow("", edges);
    waitKey(0);

    find_lines(edges);
}