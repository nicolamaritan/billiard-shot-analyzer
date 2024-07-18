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
    //void draw_initial_minimap(const std::vector<cv::Point> &balls_pos, const cv::Mat &src, cv::Mat &dst);
    //void draw_minimap(const std::vector<cv::Point> &old_balls_pos, const std::vector<cv::Point> &balls_pos, const cv::Mat &src, cv::Mat &trajectories, cv::Mat &dst);
    bool is_rectangular_pool_table(const std::vector<cv::Point>& pool_corners);
    void sort_corners_for_minimap(const std::vector<cv::Point> &corners_src, std::vector<cv::Point> &corners_dst);
private:
    const std::vector<cv::Point2f> corners_minimap = {cv::Point2f(75, 510), cv::Point2f(75, 55), cv::Point2f(935, 55), cv::Point2f(935, 510)};
    std::vector<cv::Point2f> corners_2f;
    cv::Mat projection_matrix;
};
#endif