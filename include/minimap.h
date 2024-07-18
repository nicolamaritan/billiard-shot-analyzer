// Author: Francesco Boscolo Meneguolo 2119969

#ifndef MINIMAP_H
#define MINIMAP_H

#include "playing_field_localization.h"
#include "balls_localization.h"

#include <opencv2/features2d.hpp>

/**
 * @brief Class for handling operations related to generating and drawing a minimap view.
 * 
 * The minimap class provides functionality to transform and draw a bird's-eye view of a playing field,
 * including placing and updating positions of balls on the minimap.
 */
class minimap
{
public:
    /**
     * @brief Constructor for the minimap class.
     *
     * Initializes the minimap object by setting up the projection matrix for perspective transformation
     * based on the provided playing field corners.
     *
     * @param playing_field A structure containing the corners of the playing field.
     * @param balls A structure containing the localization information of balls on the playing field.
     */
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
    const std::vector<cv::Point2f> corners_minimap = {cv::Point2f(75, 520), cv::Point2f(75, 55), cv::Point2f(940, 55), cv::Point2f(940, 520)};
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

};
#endif