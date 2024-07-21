// Author: Eddie Carraro 2121248

#ifndef BOUNDING_BOXES_DRAWER_H
#define BOUNDING_BOXES_DRAWER_H

#include "balls_localization.h"
#include "playing_field_localization.h"

/**
 * @brief Classes that draws the bounding boxes around detected balls.
 */
class bounding_boxes_drawer
{
public:
    /**
     * @brief Constructor for bounding_boxes_drawer, initializes the object with playing field and ball localization data,
     *        and determines the indices of various types of balls within the multi-tracker bounding boxes.
     *
     * @param plf_localization Playing field localization.
     * @param blls_localization Balls localization.
     * @param tracker_bboxes Vector of bounding boxes from the multi-tracker.
     */
    bounding_boxes_drawer(const playing_field_localization &plf_localization, const balls_localization &blls_localization, const std::vector<cv::Rect2d> &tracker_bboxes);

    /**
     * @brief Draws bounding boxes and lines on the given frame.
     *
     * This function draws bounding boxes around specified objects (cue ball, black ball,
     * solids, and stripes) with different colors and a specified transparency on the
     * provided frame. It also draws yellow lines along the corners of the playing field.
     *
     * @param frame The input image on which to draw the bounding boxes.
     * @param dst The output image with the drawn bounding boxes and lines.
     * @param updated_balls_bboxes A vector containing the bounding boxes for the objects.
     */
    void draw(const cv::Mat &frame, cv::Mat &dst, const std::vector<cv::Rect2d> &updated_balls_bboxes);

private:
    /**
     * @brief Draws a transparent rectangle on an image.
     *
     * This function draws a rectangle with the specified color and transparency on the
     * provided image. It first draws a filled rectangle with transparency, then draws
     * the border of the rectangle with the same color.
     *
     * @param image The image on which to draw the rectangle.
     * @param rect The rectangle to be drawn.
     * @param color The color of the rectangle.
     * @param alpha The transparency factor of the rectangle.
     */
    void draw_transparent_rect(cv::Mat &image, cv::Rect rect, cv::Scalar color, double alpha);

    // Indeces of the objects tracked by the multitracker, they are constant during multitracker life time.
    int cue_index;                              // Cue ball index.
    int black_index;                            // Black ball index.
    std::vector<int> solids_indeces;            // Solid balls indeces.
    std::vector<int> stripes_indeces;           // Stripe balls indeces.
    playing_field_localization playing_field;   // Playing field localization
    balls_localization balls;                   // Balls localization
};

#endif