#ifndef FRAME_SEGMENTATION_H
#define FRAME_SEGMENTATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

enum label_id
{
    background,
    cue,
    black,
    solids,
    stripes,
    playing_field
};

/**
 * @brief Generates the segmentation of the playing field and balls in a frame.
 *
 * This function localizes the playing field and balls in the input frame, and generates a segmentation mask where
 * different labels represent different objects (e.g., playing field, cue ball, solids, stripes).
 *
 * @param src The source image/frame to be processed.
 * @param dst The destination image/frame where the segmentation result will be stored.
 */
void get_colored_frame_segmentation(const cv::Mat& src, cv::Mat& dst, bool preserve_background);

/**
 * @brief Generates and colors the segmentation of the playing field and balls in a frame.
 *
 * This function creates a colored segmentation of the playing field and balls in the input frame.
 * It optionally preserves the background and draws yellow lines around the playing field's corners.
 *
 * @param src The source image/frame to be processed.
 * @param dst The destination image/frame where the colored segmentation result will be stored.
 * @param preserve_background Boolean flag to indicate if the background should be preserved in the segmentation.
 */
void get_frame_segmentation(const cv::Mat &src, cv::Mat &dst);

#endif