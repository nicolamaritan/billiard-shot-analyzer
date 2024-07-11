#ifndef BALLS_LOCALIZER
#define BALLS_LOCALIZER

#include "playing_field_localizer.h"

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
     * Localize the pool table.
     *
     * @param src The input image.
     */
    void localize(const cv::Mat &src);

    std::vector<cv::Rect> get_rois() { return rois; };

private:
    void localize_red_balls(const cv::Mat &src, cv::Mat &dst);
    void circles_masks(const std::vector<cv::Vec3f> &circles, std::vector<cv::Mat> &masks, cv::Size mask_size);
    void filter_empty_circles(std::vector<cv::Vec3f> &circles, const std::vector<cv::Mat> &masks, const cv::Mat &segmentation_mask, float intersection_threshold);
    void filter_out_of_bound_circles(std::vector<cv::Vec3f> &circles, const cv::Mat &table_mask, int distance_threshold);
    void filter_near_holes_circles(std::vector<cv::Vec3f> &circles, const std::vector<cv::Point> &holes_points, float distance_threshold);
    void extract_bounding_boxes(const std::vector<cv::Vec3f> &circles, std::vector<cv::Rect> &bounding_boxes);
    void fill_small_holes(cv::Mat &binary_mask, double area_threshold);
    void extract_seed_points(const cv::Mat &inrange_segmentation_mask, std::vector<cv::Point> &seed_points);
    float get_white_percentage_in_circle(const cv::Mat &src, cv::Vec3f circle);
    void filter_close_dissimilar_circles(std::vector<cv::Vec3f> &circles, float neighborhood_threshold, float distance_threshold, float radius_threshold);
    void draw_circles(const cv::Mat& src, cv::Mat& dst, std::vector<cv::Vec3f>& circles);

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
    std::vector<cv::Rect> rois;
};

#endif