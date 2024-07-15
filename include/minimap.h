#ifndef MINIMAP_H
#define MINIMAP_H

#include <opencv2/features2d.hpp>
#include "playing_field_localization.h"
#include "balls_localization.h"

class minimap
{
public:
    minimap(playing_field_localization playing_field, balls_localization balls);

    void draw_dashed_line(cv::Mat &img, cv::Point pt1, cv::Point pt2,
                          cv::Scalar color, int thickness, std::string style,
                          int gap);
    void get_balls_centers(const std::vector<cv::Rect2d>& bounding_boxes, std::vector<cv::Point>& balls_centers);
    void draw_initial_minimap(const std::vector<cv::Point> &balls_pos, const cv::Mat &src, cv::Mat &dst);
    void draw_minimap(const std::vector<cv::Point> &old_balls_pos, const std::vector<cv::Point> &balls_pos, const cv::Mat &src, cv::Mat &trajectories, cv::Mat &dst);

private:
    const std::vector<cv::Point2f> corners_minimap = {cv::Point2f(70, 60), cv::Point2f(924, 60), cv::Point2f(924, 500), cv::Point2f(70, 500)};
    std::vector<cv::Point2f> corners_2f;
    cv::Mat projection_matrix;
};
#endif