// Author: Eddie Carraro 2121248

#ifndef BOUNDING_BOXES_DRAWER
#define BOUNDING_BOXES_DRAWER

#include "balls_localization.h"
#include "playing_field_localization.h"

/**
 * @brief Classes that draws the bounding boxes around detected balls.
 */
class bounding_boxes_drawer
{
public:
    bounding_boxes_drawer(const playing_field_localization &plf_localization, const balls_localization &blls_localization, const std::vector<cv::Rect2d> &tracker_bboxes);
    void draw(const cv::Mat &frame, cv::Mat &dst, const std::vector<cv::Rect2d> &updated_balls_bboxes);

private:
    void draw_transparent_rect(cv::Mat &image, cv::Rect rect, cv::Scalar color, double alpha);

    // Indeces of the objects tracked by the multitracker, they are constant during multitracker life time.
    int cue_index;                    // Cue ball index.
    int black_index;                  // Black ball index.
    std::vector<int> solids_indeces;  // Solid balls indeces.
    std::vector<int> stripes_indeces; // Stripe balls indeces.
    playing_field_localization playing_field;
    balls_localization balls;
};

#endif