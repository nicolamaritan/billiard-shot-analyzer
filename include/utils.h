#ifndef UTILS
#define UTILS

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


cv::Vec3b get_board_color(const cv::Mat &src, float radius);
double angle_between_lines(double m1, double m2);

#endif