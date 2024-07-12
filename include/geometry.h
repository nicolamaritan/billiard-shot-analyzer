#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void intersections(const std::vector<cv::Vec3f> &lines, std::vector<cv::Point> &out_intersections, int rows, int cols);
void get_pairs_points_per_line(const std::vector<cv::Vec3f> &lines, std::vector<std::pair<cv::Point, cv::Point>> &pts);
bool intersection(std::pair<cv::Point, cv::Point> pts_line_1, std::pair<cv::Point, cv::Point> pts_line_2, cv::Point &intersection_pt, int rows, int cols);
bool is_within_image(const cv::Point &p, int rows, int cols);
double angular_coefficient(const std::pair<cv::Point, cv::Point> line);
double angle_between_lines(const std::pair<cv::Point, cv::Point> line_1, const std::pair<cv::Point, cv::Point> line_2);
bool is_vertical_line(const std::pair<cv::Point, cv::Point> line);
bool are_parallel_lines(const std::pair<cv::Point, cv::Point> line_1, const std::pair<cv::Point, cv::Point> line_2);
double intercept(const std::pair<cv::Point, cv::Point> line);

#endif