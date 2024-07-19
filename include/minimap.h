// Author: Francesco Boscolo Meneguolo 2119969

#ifndef MINIMAP_H
#define MINIMAP_H

#include "playing_field_localization.h"
#include "balls_localization.h"

#include <opencv2/features2d.hpp>

/**
 * @brief Class for handling operations related to generating and drawing a minimap view.
 *
 * The minimap class provides functionality to transform and draw a top down view of a playing field,
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
     * @param playing_field A structure localization information of the playing field.
     * @param balls A structure containing the localization information of balls on the playing field.
     * @param tracker_bboxes Vector containing the tracker bounding boxes of the initial detection.
     */
    minimap(const playing_field_localization &plf_localization, const balls_localization &blls_localization, const std::vector<cv::Rect2d> &tracker_bboxes);

    /**
     * @brief Draws the initial minimap with the positions of the balls.
     * @param dst The destination image where the minimap will be drawn.
     */
    void draw_initial_minimap(cv::Mat &dst);
    
    /**
     * @brief Draws the minimap, updating ball positions and their trajectories.
     * @param dst The destination image where the minimap will be drawn.
     */
    void draw_minimap(cv::Mat &dst);

    void update(const std::vector<cv::Rect2d> &updated_balls_bboxes);

private:
    /**
     * @brief Computes the positions of the balls based on their bounding boxes.
     * @param bounding_boxes The bounding boxes of the balls.
     * @param balls_pos The computed positions of the balls.
     */
    void get_balls_pos(const std::vector<cv::Rect2d> &bounding_boxes, std::vector<cv::Point> &balls_centers);
    
    void load_balls_indeces(const std::vector<cv::Point> &balls_pos);

    /**
     * @brief Checks if the pool table is rectangular based on the given corners.
     * @param pool_corners The corners of the pool table.
     * @return True if the pool table is rectangular, false otherwise.
     */
    bool is_rectangular_pool_table(const std::vector<cv::Point> &pool_corners);
    
    bool is_inside_playing_field(const cv::Point2f ball_position);
    
    bool is_inside_hole(const cv::Point2f ball_position);

    /**
     * @brief Sorts the corners of the playing field for the minimap.
     * @param corners_src The source corners of the playing field.
     * @param corners_dst The destination corners sorted for the minimap.
     */
    void sort_corners_for_minimap(const std::vector<cv::Point> &corners_src, std::vector<cv::Point> &corners_dst);

    /**
     * @brief Draws a dotted line on the given image.
     * @param img The image on which to draw the line.
     * @param pt1 The starting point of the line.
     * @param pt2 The ending point of the line.
     * @param color The color of the line.
     * @param thickness The thickness of the line.
     * @param gap The gap between the segments of the line.
     */
    void draw_dotted_line(cv::Mat &img, cv::Point pt1, cv::Point pt2, cv::Scalar color, int thickness, int gap);

    const int X1 = 75;
    const int X2 = 940;
    const int Y1 = 59;
    const int Y2 = 520;
    const std::vector<cv::Point2f> corners_minimap = {cv::Point2f(X1, Y2), cv::Point2f(X1, Y1), cv::Point2f(X2, Y1), cv::Point2f(X2, Y2)}; // These coordinates represent the corners of the minimap where the playing field will be projected.
    const cv::Point2f UPPER_LEFT_HOLE = cv::Point2f(83, 73);
    const cv::Point2f UPPER_MIDDLE_HOLE = cv::Point2f(510, 63);
    const cv::Point2f UPPER_RIGHT_HOLE = cv::Point2f(940, 73);
    const cv::Point2f BOTTOM_LEFT_HOLE = cv::Point2f(83, 505);
    const cv::Point2f BOTTOM_MIDDLE_HOLE = cv::Point2f(510, 513);
    const cv::Point2f BOTTOM_RIGHT_HOLE = cv::Point2f(949, 505);
    const int HOLE_RADIUS = 17;
    const int BALL_RADIUS = 10;                                     // Radius of the balls drawn on the minimap.
    const int THICKNESS = 2;                                        // Thickness of the contour lines drawn around the balls on the minimap.
    const cv::Scalar SOLID_BALL_COLOR = cv::Scalar(255, 180, 160);  // Color used for solid balls.
    const cv::Scalar STRIPE_BALL_COLOR = cv::Scalar(179, 179, 255); // Color used for striped balls.
    const cv::Scalar BLACK_BALL_COLOR = cv::Scalar(70, 70, 70);     // Color used for the black ball.
    const cv::Scalar CUE_BALL_COLOR = cv::Scalar(255, 255, 255);    // Color used for the cue ball.
    const cv::Scalar CONTOUR_COLOR = cv::Scalar(0, 0, 0);           // Color used for the contours of the balls.
    const int GAP = 10;                                             // Gap size for dashed lines.
    const cv::Point2f INVALID_POSITION = cv::Point2f(0, 0);
    const std::string IMAGES_DIRECTORY = "images";
    const std::string MINIMAP_IMAGE_FILENAME = "pool_table.png";
    std::vector<cv::Point2f> corners_2f;
    cv::Mat projection_matrix;
    playing_field_localization playing_field;
    balls_localization balls;
    int cue_index;
    int black_index;
    std::vector<int> solids_indeces;
    std::vector<int> stripes_indeces;
    std::vector<cv::Point> current_balls_pos;
    std::vector<cv::Point> old_balls_pos;
    cv::Mat empty_minimap;
    cv::Mat trajectories;
};
#endif