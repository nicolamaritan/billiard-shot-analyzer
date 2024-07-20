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
     * @param plf_localization A structure localization information of the playing field.
     * @param blls_localization A structure containing the localization information of balls on the playing field.
     * @param tracker_bboxes Vector containing the tracker bounding boxes of the initial detection.
     */
    minimap(const playing_field_localization &plf_localization, const balls_localization &blls_localization, const std::vector<cv::Rect2d> &tracker_bboxes);

    /**
     * @brief Draws the initial minimap with the positions of the balls.
     * 
     * @param dst The destination image where the minimap will be drawn.
     */
    void draw_initial_minimap(cv::Mat &dst);
    
    /**
     * @brief Draws the minimap, updating ball positions and their trajectories.
     * 
     * @param dst The destination image where the minimap will be drawn.
     */
    void draw_minimap(cv::Mat &dst);

    /**
     * @brief Updates the positions of the balls.
     * 
     * This method updates the current ball positions based on the provided bounding boxes.
     * 
     * @param updated_balls_bboxes Vector of rectangles representing the current updated bounding boxes of the balls.
    */
    void update(const std::vector<cv::Rect2d> &updated_balls_bboxes);

private:
    /**
     * @brief Computes the positions of the balls based on their bounding boxes.
     * 
     * @param bounding_boxes The bounding boxes of the balls.
     * @param balls_pos The computed positions of the balls.
     */
    void get_balls_pos(const std::vector<cv::Rect2d> &bounding_boxes, std::vector<cv::Point> &balls_pos);
    
    /**
     * @brief Loads the indices of the balls.
     * 
     * This method loads the indices of the balls into the respective vectors for solids, stripes, black, and cue balls.
     * 
     * @param balls_pos Vector of points representing the positions of the balls.
    */
    void load_balls_indeces(const std::vector<cv::Point> &balls_pos);

    /**
     * @brief Checks if the pool table is rectangular based on the given corners.
     * @param pool_corners The corners of the pool table.
     * @return True if the pool table is rectangular, false otherwise.
     */
    bool is_rectangular_pool_table(const std::vector<cv::Point> &pool_corners);
    
    /**
     * @brief Checks if a ball is inside the playing field.
     * 
     * This method checks if a given ball position is inside the playing field.
     * 
     * @param ball_position The position of the ball to check.
     * @return True if the ball is inside the playing field, false otherwise.
    */
    bool is_inside_playing_field(const cv::Point2f ball_position);
    
    /**
     * @brief Checks if a ball is inside one of the pool table's holes.
     * 
     * This method checks if a given ball position is inside one of the pool table's holes.
     * 
     * @param ball_position The position of the ball to check.
     * @return True if the ball is inside a hole, false otherwise.
    */
    bool is_inside_hole(const cv::Point2f ball_position);

    /**
     * @brief Sorts the corners of the playing field in order to match them with the corners of the minimap.
     * @param original_corners The unsorted corners of the playing field.
     * @param sorted_corners The corners sorted for the minimap.
     */
    void sort_corners_for_minimap(const std::vector<cv::Point> &original_corners, std::vector<cv::Point> &sorted_corners);

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

    const int X1 = 75; // x-coordinate of left corners.
    const int X2 = 940; // x-coordinate of right corners.
    const int Y1 = 59; // y-coordinate of upper corners.
    const int Y2 = 520; // y-coordinate of bottom corners.
    const std::vector<cv::Point2f> corners_minimap = {cv::Point2f(X1, Y2), cv::Point2f(X1, Y1), cv::Point2f(X2, Y1), cv::Point2f(X2, Y2)}; // These coordinates represent the corners of the minimap where the playing field will be projected.
    const cv::Point2f UPPER_LEFT_HOLE = cv::Point2f(83, 73);        // Upper left hole position.
    const cv::Point2f UPPER_MIDDLE_HOLE = cv::Point2f(510, 63);     // Upper middle hole position.
    const cv::Point2f UPPER_RIGHT_HOLE = cv::Point2f(940, 73);      // Upper right hole position.
    const cv::Point2f BOTTOM_LEFT_HOLE = cv::Point2f(83, 505);      // bottom left hole position.
    const cv::Point2f BOTTOM_MIDDLE_HOLE = cv::Point2f(510, 513);   // bottom middle hole position.
    const cv::Point2f BOTTOM_RIGHT_HOLE = cv::Point2f(949, 505);    // bottom right hole position.
    const int HOLE_RADIUS = 17;                                     // Hole radius.
    const int BALL_RADIUS = 10;                                     // Radius of the balls drawn on the minimap.
    const int THICKNESS = 2;                                        // Thickness of the contour lines drawn around the balls on the minimap.
    const cv::Scalar SOLID_BALL_COLOR = cv::Scalar(255, 180, 160);  // Color used for solid balls.
    const cv::Scalar STRIPE_BALL_COLOR = cv::Scalar(179, 179, 255); // Color used for striped balls.
    const cv::Scalar BLACK_BALL_COLOR = cv::Scalar(70, 70, 70);     // Color used for the black ball.
    const cv::Scalar CUE_BALL_COLOR = cv::Scalar(255, 255, 255);    // Color used for the cue ball.
    const cv::Scalar CONTOUR_COLOR = cv::Scalar(0, 0, 0);           // Color used for the contours of the balls.
    const int GAP = 10;                                             // Gap size for dashed lines.
    const cv::Point2f INVALID_POSITION = cv::Point2f(0, 0);         // Invalid position returned by the multitracker when a ball disappears (it goes inside a hole).
    const std::string IMAGES_DIRECTORY = "images";                  // Name of the directory.
    const std::string MINIMAP_IMAGE_FILENAME = "pool_table.png";    // Name of the file used to build the minimap.
    std::vector<cv::Point2f> corners_2f;                            // Vector of corners used for the conversion from Point to Point2f.
    cv::Mat projection_matrix;                                      // Projection matrix that describes the transformation from the input frame to the minimap image.
    playing_field_localization playing_field;                       // Encapsulates playing field.
    balls_localization balls;                                       // Contains the localizations of the balls in the first video frame.
    std::vector<cv::Point> current_balls_pos;                       // Positions of the balls in the current frame.
    std::vector<cv::Point> old_balls_pos;                           // Positions of the balls in the previous frame, i.e. the frame before the current frame.
    cv::Mat empty_minimap;                                          // Image that contains the empty minimap.
    cv::Mat trajectories;                                           // Image that contains the trajectories of the balls.
    
    // Indeces of the objects tracked by the multitracker, they are constant during multitracker life time.
    int cue_index;                                                  // Cue ball index.                                                  
    int black_index;                                                // Black ball index.
    std::vector<int> solids_indeces;                                // Solid balls indeces.
    std::vector<int> stripes_indeces;                               // Stripe balls indeces.
};
#endif