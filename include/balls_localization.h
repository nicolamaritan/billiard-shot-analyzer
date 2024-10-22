// Author: Nicola Maritan 2121717

#ifndef BALLS_LOCALIZATION_H
#define BALLS_LOCALIZATION_H

#include "playing_field_localization.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief Structure to hold the localization information of a single ball.
 */
struct ball_localization
{
    cv::Vec3f circle;      // Circle representation as (x, y, radius).
    cv::Rect bounding_box; // Bounding box for the ball.
    float confidence;      // Confidence score of the localization.
};
typedef struct ball_localization ball_localization;

/**
 * @brief Equality operator for ball_localization struct.
 */
bool operator==(const ball_localization &lhs, const ball_localization &rhs);

/**
 * @brief Inequality operator for ball_localization struct.
 */
bool operator!=(const ball_localization &lhs, const ball_localization &rhs);

/**
 * @brief
 */
const ball_localization NO_LOCALIZATION = {cv::Vec3f(-1, -1, -1), cv::Rect(-1, -1, -1, -1), 0.0};

/**
 * @brief Structure to hold the localization information for all balls.
 */
struct balls_localization
{
    std::vector<ball_localization> solids;  // Localizations of solid balls.
    std::vector<ball_localization> stripes; // Localizations of striped balls.
    ball_localization black;                // Localization of the black ball.
    ball_localization cue;                  // Localization of the cue ball.
};
typedef struct balls_localization balls_localization;

/**
 * @brief Class for localizing balls on a playing field.
 */
class balls_localizer
{
public:
    /**
     * @brief Constructor for balls_localizer.
     *
     * @param localization The localization of the playing field.
     */
    balls_localizer(const playing_field_localization &localization)
        : playing_field{localization} {};
    /**
     * Localize the balls.
     *
     * @param src The input image.
     */
    void localize(const cv::Mat &src);

    /**
     * Returns the bounding boxes of all detected balls.
     *
     * @return the bounding boxes of all detected balls.
     */
    std::vector<cv::Rect> get_bounding_boxes() { return bounding_boxes; };

    /**
     * Returns the obtained localization.
     *
     * @return the obtained localization.
     */
    balls_localization get_localization() { return localization; }

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
    void get_bounding_boxes(const std::vector<cv::Vec3f> &circles, std::vector<cv::Rect> &bounding_boxes);

    /**
     * @brief Computes the bounding box for a given circle.
     *
     * This function calculates the bounding box for a circle represented by a Vec3f (x, y, radius).
     * The bounding box is a rectangle that fully contains the circle.
     *
     * @param circle A Vec3f representing a circle (x, y, radius).
     * @return cv::Rect The bounding box that fully contains the specified circle.
     */
    cv::Rect get_bounding_box(cv::Vec3f circle);

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
     * @brief Computes the ratio of white pixels within a specified circle on the source image for looking for the cue ball.
     *
     * This function calculates the ratio of white pixels within a given circle, intersected with the provided segmentation mask.
     * It uses a specific range in the HSV color space to determine the white pixels.
     *
     * @param src The source image in HSV color space.
     * @param segmentation_mask The segmentation mask used to exclude certain areas.
     * @param circle A Vec3f representing a circle (x, y, radius) where the calculation is performed.
     * @return float The ratio of white pixels within the specified circle.
     */
    float get_white_ratio_in_circle_cue(const cv::Mat &src, const cv::Mat &segmentation_mask, cv::Vec3f circle);

    /**
     * @brief Computes the ratio of black pixels within a specified circle on the source image for looking for the black ball.
     *
     * This function calculates the ratio of black pixels within a given circle, intersected with the provided segmentation mask.
     * It uses a specific range in the HSV color space to determine the black pixels.
     *
     * @param src The source image in HSV color space.
     * @param segmentation_mask The segmentation mask used to exclude certain areas.
     * @param circle A Vec3f representing a circle (x, y, radius) where the calculation is performed.
     * @return float The ratio of black pixels within the specified circle.
     */
    float get_black_ratio_in_circle(const cv::Mat &src, const cv::Mat &segmentation_mask, cv::Vec3f circle);

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
     * @brief Computes the distance of the mean hue value within a circular region from the middle hue value (128).
     *
     * It calculates the absolute difference between the mean hue
     * and the middle hue value (128).
     *
     * @param src Input source image in BGR format.
     * @param segmentation_mask Binary mask where ball regions are masked out.
     * @param circle A Vec3f representing the circle (x, y, radius).
     * @return The absolute distance of the mean hue value from 128.
     *
     */
    float distance_from_middle_hue(const cv::Mat &src, const cv::Mat &segmentation_mask, cv::Vec3f circle);

    /**
     * @brief Removes connected components from the mask that have a diameter smaller than a specified minimum.
     *
     * This function iterates through connected components in the mask, computes the minimum enclosing circle for each component,
     * and removes components whose diameter is less than the specified minimum diameter.
     *
     * @param mask The binary mask from which small connected components will be removed.
     * @param min_diameter The minimum diameter threshold. Components with a diameter smaller than this value will be removed.
     */
    void remove_connected_components_by_diameter(cv::Mat &mask, double min_diameter);

    /**
     * @brief Computes the ratio of white pixels within a specified circle on the source image for looking for striped balls.
     *
     * This function calculates the ratio of white pixels within a given circle, intersected with the provided segmentation mask.
     * It uses a specific range in the HSV color space to determine the white pixels.
     *
     * @param src The source image in HSV color space.
     * @param segmentation_mask The segmentation mask used to exclude certain areas.
     * @param circle A Vec3f representing a circle (x, y, radius) where the calculation is performed.
     * @return float The ratio of white pixels within the specified circle.
     */
    float get_white_ratio_in_circle_stripes(const cv::Mat &src, const cv::Mat &segmentation_mask, cv::Vec3f circle);

    /**
     * @brief Identifies and localizes the cue ball in the given image.
     *
     * This function finds the cue ball by computing the white pixel ratio within each circle.
     * The circle with the highest ratio is considered the cue ball.
     *
     * @param src The source image.
     * @param segmentation_mask The segmentation mask used to filter relevant areas.
     * @param circles A vector containing circles detected, where each circle is represented by a Vec3f (x, y, radius).
     */
    void find_cue_ball(const cv::Mat &src, const cv::Mat &segmentation_mask, const std::vector<cv::Vec3f> &circles);

    /**
     * @brief Identifies and localizes the black ball in the given image.
     *
     * This function finds the black ball by computing the black pixel ratio within each circle.
     * The circle with the highest ratio is considered the black ball.
     *
     * @param src The source image.
     * @param segmentation_mask The segmentation mask used to filter relevant areas.
     * @param circles A vector containing circles detected, where each circle is represented by a Vec3f (x, y, radius).
     */
    void find_black_ball(const cv::Mat &src, const cv::Mat &segmentation_mask, const std::vector<cv::Vec3f> &circles);

    /**
     * @brief Identifies and localizes the stripe balls in the given image.
     *
     * This function finds the stripe balls by computing the white pixel ratio within each circle
     * and filtering out circles that are likely cue or black balls.
     *
     * @param src The source image.
     * @param segmentation_mask The segmentation mask used to filter relevant areas.
     * @param circles A vector containing circles detected, where each circle is represented by a Vec3f (x, y, radius).
     */
    void find_stripe_balls(const cv::Mat &src, const cv::Mat &segmentation_mask, const std::vector<cv::Vec3f> &circles);

    /**
     * @brief Identifies and localizes the solid balls in the given image.
     *
     * This function identifies solid balls by excluding the cue, black, and stripe balls from the detected circles.
     * The confidence for solid balls is estimated based on the confidences of other classified balls.
     *
     * @param src The source image.
     * @param segmentation_mask The segmentation mask used to filter relevant areas.
     * @param circles A vector containing circles detected, where each circle is represented by a Vec3f (x, y, radius).
     */
    void find_solid_balls(const cv::Mat &src, const cv::Mat &segmentation_mask, const std::vector<cv::Vec3f> &circles);

    /**
     * @brief Generates a mask of the intersection between a circular region and the negated ball segmentation mask.
     *
     * @param segmentation_mask Binary mask where ball regions are masked out.
     * @param circle A Vec3f representing a circle (x, y, radius).
     * @param mask Output binary mask representing the intersection area.
     */
    void get_circle_and_field_mask(const cv::Mat &segmentation_mask, cv::Vec3f circle, cv::Mat &mask);

    const float BOUNDING_BOX_RESCALE = 1.2;         // A scaling factor to rescale bounding boxes for better tracking.
    const float MAX_SIZE_BOUNDING_BOX_RESCALE = 14; // The maximum size limit for bounding box rescaling.

    const playing_field_localization playing_field; //  An instance of playing_field_localization, which represents the playing field's localization data.
    std::vector<cv::Rect> bounding_boxes;           // A vector of cv::Rect objects storing the bounding boxes of detected balls.
    balls_localization localization;                // An instance of balls_localization, which contains the localization data of detected balls.
};

#endif