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

    std::vector<cv::Mat> hsv_channels;
    cv::split(src_hsv, hsv_channels);
    int desiredValue = 255;
    hsv_channels[2].setTo(desiredValue);
    cv::merge(hsv_channels, src_hsv);

    imshow("", src_hsv);
    waitKey(0);

    Mat data;
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // do kmeans
    Mat labels, centers;
    kmeans(data, 3, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 1.0), 3,
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

vector<Vec2f> playing_field_localizer::find_lines(const cv::Mat &edges)
{
    Mat cdst;

    // Copy edges to the images that will display the results in BGR
    cvtColor(edges, cdst, COLOR_GRAY2BGR);
    // Standard Hough Line Transform
    vector<Vec2f> lines;                                         // will hold the results of the detection
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
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
    }

    imshow("", cdst);
    waitKey();

    return lines;
}

void playing_field_localizer::localize(const Mat &src, Mat &dst)
{
    GaussianBlur(src.clone(), src, Size(3, 3), 15, 15);

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

    non_maxima_connected_component_suppression(mask.clone(), mask);
    imshow("", mask);
    waitKey(0);

    Mat edges;
    Canny(mask, edges, 50, 150);
    imshow("", edges);
    waitKey(0);

    vector<Vec2f> lines = find_lines(edges);
    vector<Vec2f> refined_lines = refine_lines(lines);

    draw_lines(edges, refined_lines);
}

vector<Vec2f> playing_field_localizer::refine_lines(vector<Vec2f> &lines)
{
    vector<Vec2f> refined_lines;
    while (!lines.empty())
    {
        Vec2f reference_line = lines.back();
        lines.pop_back();
        vector<Vec2f> similar_lines;

        dump_similar_lines(reference_line, lines, similar_lines);

        Vec2f mean_line;
        for (auto similar_line : similar_lines)
        {
            mean_line += similar_line;
        }
        mean_line[0] /= similar_lines.size();
        mean_line[1] /= similar_lines.size();
        refined_lines.push_back(mean_line);
    }
    return refined_lines;
}

void playing_field_localizer::draw_lines(const Mat &src, const vector<cv::Vec2f> &lines)
{
    Mat src_bgr;
    cvtColor(src, src_bgr, COLOR_GRAY2BGR);

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
        line(src_bgr, pt1, pt2, Scalar(0, 255, 0), 1, LINE_AA);
    }

    imshow("", src_bgr);
    waitKey();
}

void playing_field_localizer::dump_similar_lines(Vec2f reference_line, vector<cv::Vec2f> &lines, vector<cv::Vec2f> &similar_lines)
{
    similar_lines.push_back(reference_line);
    // Insert into similar_lines all the similar lines and removes them from lines.
    int i = 0;
    while (i < lines.size())
    {
        Vec2f line = lines.at(i);
        if (abs(line[0] - reference_line[0]) < 25 && abs(line[1] - reference_line[1]) < 0.2)
        {
            similar_lines.push_back(line);
            lines.erase(lines.begin() + i);
        }
        else
        {
            i++;
        }
    }
}

void playing_field_localizer::non_maxima_connected_component_suppression(const cv::Mat &src, cv::Mat &dst)
{
    src.copyTo(dst);
    Mat connected_components_labels, stats, centroids;
    connectedComponentsWithStats(src, connected_components_labels, stats, centroids);

    int max_label_component = 1;
    int max_area = 0;
    for (int i = 1; i < stats.rows; i++)
    {
        int component_area = stats.at<int>(i, 4);
        if (component_area > max_area)
        {
            max_area = component_area;
            max_label_component = i;
        }
    }

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
