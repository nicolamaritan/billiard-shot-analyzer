#ifndef UTILS_H
#define UTILS_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void intersections(const std::vector<cv::Vec3f> &lines, std::vector<cv::Point> &out_intersections, int rows, int cols);
void get_pairs_points_per_line(const std::vector<cv::Vec3f> &lines, std::vector<std::pair<cv::Point, cv::Point>> &pts);
bool intersection(std::pair<cv::Point, cv::Point> pts_line_1, std::pair<cv::Point, cv::Point> pts_line_2, cv::Point &intersection_pt, int rows, int cols);
bool is_within_image(const cv::Point &p, int rows, int cols);

#endif