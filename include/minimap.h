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

    /**
     * @brief Draws a dashed line on the given image.
     * @param img The image on which to draw the line.
     * @param pt1 The starting point of the line.
     * @param pt2 The ending point of the line.
     * @param color The color of the line.
     * @param thickness The thickness of the line.
     * @param style The style of the line, either "dotted" or "dashed".
     * @param gap The gap between the segments of the line.
     */
    void draw_dashed_line(cv::Mat &img, cv::Point pt1, cv::Point pt2,
                          cv::Scalar color, int thickness, std::string style,
                          int gap);

    /**
     * @brief Computes the positions of the balls based on their bounding boxes.
     * @param bounding_boxes The bounding boxes of the balls.
     * @param balls_pos The computed positions of the balls.
     */
    void get_balls_pos(const std::vector<cv::Rect2d>& bounding_boxes, std::vector<cv::Point>& balls_centers);

    /**
     * @brief Draws the initial minimap with the positions of the balls.
     * @param balls_pos The positions of the balls.
     * @param balls The balls localization data.
     * @param solids_indeces The indices of solid balls.
     * @param stripes_indeces The indices of stripe balls.
     * @param black_index The index of the black ball.
     * @param cue_index The index of the cue ball.
     * @param src The source image.
     * @param dst The destination image where the minimap will be drawn.
     */
    void draw_initial_minimap(const std::vector<cv::Point> &balls_pos, const balls_localization &balls, std::vector<int> &solids_indeces, std::vector<int> &stripes_indeces, int &black_index, int &cue_index, const cv::Mat &src, cv::Mat &dst);

    /**
     * @brief Draws the minimap, updating ball positions and their trajectories.
     * @param old_balls_pos The previous positions of the balls.
     * @param balls_pos The current positions of the balls.
     * @param solids_indeces The indices of solid balls.
     * @param stripes_indeces The indices of stripe balls.
     * @param black_index The index of the black ball.
     * @param cue_index The index of the cue ball.
     * @param src The source image.
     * @param trajectories The image showing ball trajectories.
     * @param dst The destination image where the minimap will be drawn.
     */
    void draw_minimap(const std::vector<cv::Point> &old_balls_pos, const std::vector<cv::Point> &balls_pos, const std::vector<int> &solids_indeces, const std::vector<int> &stripes_indeces, const int black_index, const int cue_index, const cv::Mat &src, cv::Mat &trajectories, cv::Mat &dst);

    //void draw_initial_minimap(const std::vector<cv::Point> &balls_pos, const cv::Mat &src, cv::Mat &dst);
    //void draw_minimap(const std::vector<cv::Point> &old_balls_pos, const std::vector<cv::Point> &balls_pos, const cv::Mat &src, cv::Mat &trajectories, cv::Mat &dst);

    /**
     * @brief Checks if the pool table is rectangular based on the given corners.
     * @param pool_corners The corners of the pool table.
     * @return True if the pool table is rectangular, false otherwise.
     */
    bool is_rectangular_pool_table(const std::vector<cv::Point>& pool_corners);

    /**
     * @brief Sorts the corners of the playing field for the minimap.
     * @param corners_src The source corners of the playing field.
     * @param corners_dst The destination corners sorted for the minimap.
     */
    void sort_corners_for_minimap(const std::vector<cv::Point> &corners_src, std::vector<cv::Point> &corners_dst);

private:

    /**
     * @brief Coordinates of the four corners of the minimap.
     * 
     * These coordinates represent the corners of the minimap where the playing field will be projected.
     */
    const std::vector<cv::Point2f> corners_minimap = {cv::Point2f(75, 520), cv::Point2f(75, 55), cv::Point2f(940, 55), cv::Point2f(940, 520)};
    
    /**
     * @brief Vector to store the corners of the playing field after sorting and transforming.
     * 
     * This vector is resized in the constructor of the minimap class and is used to store the
     * transformed corners of the playing field for use in perspective transformation.
     */
    std::vector<cv::Point2f> corners_2f;
    
    /**
     * @brief Matrix to store the perspective transformation matrix.
     * 
     * This matrix is computed using cv::getPerspectiveTransform to map the playing field coordinates
     * to the minimap coordinates.
     */
    cv::Mat projection_matrix;
    
    const int BALL_RADIUS = 10;     // Radius of the balls drawn on the minimap.
    
    const int THICKNESS = 2;        // Thickness of the contour lines drawn around the balls on the minimap.
    
    const cv::Scalar SOLID_BALL_COLOR = cv::Scalar(255, 180, 160);  // Color used for solid balls.
    
    const cv::Scalar STRIPE_BALL_COLOR = cv::Scalar(179, 179, 255); // Color used for striped balls.
    
    const cv::Scalar BLACK_BALL_COLOR = cv::Scalar(70, 70, 70);     // Color used for the black ball.
    
    const cv::Scalar CUE_BALL_COLOR = cv::Scalar(255, 255, 255);    // Color used for the cue ball.
    
    const cv::Scalar CONTOUR_COLOR = cv::Scalar(0, 0, 0);       // Color used for the contours of the balls.
    
    const int GAP = 10;         // Gap size for dashed lines.

};
#endif