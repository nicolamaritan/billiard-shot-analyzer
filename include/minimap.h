// Author: Francesco Boscolo Meneguolo 2119969

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
    void get_balls_pos(const std::vector<cv::Rect2d>& bounding_boxes, std::vector<cv::Point>& balls_centers);
    void draw_initial_minimap(const std::vector<cv::Point> &balls_pos, const balls_localization &balls, std::vector<int> &solids_indeces, std::vector<int> &stripes_indeces, int &black_index, int &cue_index, const cv::Mat &src, cv::Mat &dst);
    void draw_minimap(const std::vector<cv::Point> &old_balls_pos, const std::vector<cv::Point> &balls_pos, const std::vector<int> &solids_indeces, const std::vector<int> &stripes_indeces, const int black_index, const int cue_index, const cv::Mat &src, cv::Mat &trajectories, cv::Mat &dst);
    bool is_rectangular_pool_table(const std::vector<cv::Point>& pool_corners);
    bool is_inside_playing_field(const cv::Point2f ball_position);
    bool is_inside_hole(const cv::Point2f ball_position);
    void sort_corners_for_minimap(const std::vector<cv::Point> &corners_src, std::vector<cv::Point> &corners_dst);
private:
    const int X1 = 75;
    const int X2 = 940;
    const int Y1 = 59;
    const int Y2 = 520;
    const std::vector<cv::Point2f> corners_minimap = {cv::Point2f(X1, Y2), cv::Point2f(X1, Y1), cv::Point2f(X2, Y1), cv::Point2f(X2, Y2)};
    const cv::Point2f UPPER_LEFT_HOLE = cv::Point2f(83, 73);
    const cv::Point2f UPPER_MIDDLE_HOLE = cv::Point2f(510, 63);
    const cv::Point2f UPPER_RIGHT_HOLE = cv::Point2f(940, 73);
    const cv::Point2f BOTTOM_LEFT_HOLE = cv::Point2f(83, 505);
    const cv::Point2f BOTTOM_MIDDLE_HOLE = cv::Point2f(510, 513);
    const cv::Point2f BOTTOM_RIGHT_HOLE = cv::Point2f(949, 505);
    const int HOLE_RADIUS = 17;
    std::vector<cv::Point2f> corners_2f;
    cv::Mat projection_matrix;
    const int BALL_RADIUS = 10;
    const int THICKNESS = 2;
    const cv::Scalar SOLID_BALL_COLOR = cv::Scalar(255, 180, 160);
    const cv::Scalar STRIPE_BALL_COLOR = cv::Scalar(179, 179, 255);
    const cv::Scalar BLACK_BALL_COLOR = cv::Scalar(70, 70, 70);
    const cv::Scalar CUE_BALL_COLOR = cv::Scalar(255, 255, 255);
    const cv::Scalar CONTOUR_COLOR = cv::Scalar(0, 0, 0);
    const int GAP = 10;
    const cv::Point2f INVALID_POSITION = cv::Point2f(0, 0);

};
#endif