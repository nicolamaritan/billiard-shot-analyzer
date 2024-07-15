#ifndef MINIMAP_H
#define MINIMAP_H

#include <opencv2/features2d.hpp>

class minimap
{
public:
    void draw_dashed_line(cv::Mat &img, cv::Point pt1, cv::Point pt2,
                          cv::Scalar color, int thickness, std::string style,
                          int gap);
    void get_balls_centers(const std::vector<cv::Rect2d>& bounding_boxes, std::vector<cv::Point>& balls_centers);
    void draw_initial_minimap(const std::vector<cv::Point> &corners, const std::vector<cv::Point> &balls_pos, const cv::Mat &src, cv::Mat &dst);
    void draw_minimap(const std::vector<cv::Point> &corners, const std::vector<cv::Point> &old_balls_pos, const std::vector<cv::Point> &balls_pos, const cv::Mat &src, cv::Mat &trajectories, cv::Mat &dst);
};
#endif