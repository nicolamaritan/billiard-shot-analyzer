#ifndef BALLS_LOCALIZATION_H
#define BALLS_LOCALIZATION_H

#include "playing_field_localization.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct ball_localization
{
    cv::Vec3f circle;
    cv::Rect bounding_box;
};
typedef struct ball_localization ball_localization;

struct balls_localization
{
    std::vector<ball_localization> solids;
    std::vector<ball_localization> stripes;
    ball_localization black;
    ball_localization cue;
};
typedef struct balls_localization balls_localization;

class balls_localizer
{
public:
    balls_localizer(const playing_field_localization &localization)
        : playing_field{localization} {};
    /**
     * Localize the balls.
     *
     * @param src The input image.
     */
    void localize(const cv::Mat &src);

    std::vector<cv::Rect> get_bounding_boxes() { return bounding_boxes; };

private:
    /**
     * @brief Generates binary masks for each circle in the input vector.
     *
     * @param circles A vector of circles for which masks are generated.
     * @param masks A vector to store the generated masks.
     * @param mask_size The size of each mask.
     */
    void circles_masks(const std::vector<cv::Vec3f> &circles, std::vector<cv::Mat> &masks, cv::Size mask_size);

    /**
     * @brief Filters out circles that do not significantly intersect with a given segmentation mask.
     *
     * @param circles A vector of circles to filter.
     * @param masks Binary masks corresponding to the circles.
     * @param segmentation_mask The segmentation mask to check intersection with.
     * @param intersection_threshold The minimum intersection ratio required to keep a circle.
     */
    void filter_empty_circles(std::vector<cv::Vec3f> &circles, const std::vector<cv::Mat> &masks, const cv::Mat &segmentation_mask, float intersection_threshold);

    /**
     * @brief Filters out circles that are outside a specified table mask.
     *
     * @param circles A vector of circles to filter.
     * @param table_mask A mask representing the valid area for circles.
     * @param distance_threshold Erosion distance for the table mask.
     */
    void filter_out_of_bound_circles(std::vector<cv::Vec3f> &circles, const cv::Mat &table_mask, int distance_threshold);

    /**
     * @brief Filters out circles that are too close to specified holes.
     *
     * @param circles A vector of circles to filter.
     * @param holes_points A vector of points representing holes.
     * @param distance_threshold Minimum distance a circle must be from a hole to be kept.
     */
    void filter_near_holes_circles(std::vector<cv::Vec3f> &circles, const std::vector<cv::Point> &holes_points, float distance_threshold);

    /**
     * @brief Extracts bounding boxes for each circle.
     *
     * @param circles A vector of circles to generate bounding boxes for.
     * @param bounding_boxes A vector to store the generated bounding boxes.
     */
    void extract_bounding_boxes(const std::vector<cv::Vec3f> &circles, std::vector<cv::Rect> &bounding_boxes);

    /**
     * @brief Fills small holes in a binary mask.
     *
     * @param binary_mask The binary mask to process.
     * @param area_threshold Maximum area of holes to fill.
     */
    void fill_small_holes(cv::Mat &binary_mask, double area_threshold);

    /**
     * @brief Extracts seed points from a given binary mask.
     *
     * @param inrange_segmentation_mask Binary mask to extract seed points from.
     * @param seed_points A vector to store the extracted seed points.
     */
    void extract_seed_points(const cv::Mat &inrange_segmentation_mask, std::vector<cv::Point> &seed_points);

    /**
     * @brief Calculates the percentage of white pixels within a given circle in an image.
     *
     * @param src The source image.
     * @param circle The circle to analyze.
     * @return The percentage of white pixels in the circle.
     */
    float get_white_percentage_in_circle(const cv::Mat &src, cv::Vec3f circle);

    /**
     * @brief Filters out circles that are close to each other but significantly different in radius and position.
     *
     * @param circles A vector of circles to filter.
     * @param neighborhood_threshold Distance threshold for neighborhood consideration.
     * @param distance_threshold Distance threshold for circle position comparison.
     * @param radius_threshold Radius difference threshold for filtering.
     */
    void filter_close_dissimilar_circles(std::vector<cv::Vec3f> &circles, float neighborhood_threshold, float distance_threshold, float radius_threshold);

    /**
     * @brief Draws circles on an image.
     *
     * @param src The source image.
     * @param dst The destination image where circles will be drawn.
     * @param circles A vector of circles to draw.
     */
    void draw_circles(const cv::Mat &src, cv::Mat &dst, std::vector<cv::Vec3f> &circles);

    /**
     * @brief Return the estimated board color.
     *
     * It computes the color of the board by considering a circle of a given radius around
     * the center of the image and picking the median value.
     *
     * @param src Input image containing the board.
     * @param radius Radius from the image center in which to compute the board color.
     * @return the computed color of the board.
     */
    cv::Vec3b get_board_color(const cv::Mat &src, float radius);

    const playing_field_localization playing_field;
    std::vector<cv::Rect> bounding_boxes;
};

#endif