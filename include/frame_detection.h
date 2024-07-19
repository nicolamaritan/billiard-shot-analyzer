// Author: Nicola Maritan 2121717
#ifndef FRAME_DETECTION_H
#define FRAME_DETECTION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief Detects and draws localized playing field and balls on a frame.
 *
 * This function processes the input frame to detect the playing field and various balls, 
 * and draws them with different colored transparent rectangles. It also draws the corners 
 * of the playing field with yellow lines.
 *
 * @param src The source image/frame to be processed.
 * @param dst The destination image/frame where the results will be drawn.
 */
void get_frame_detection(const cv::Mat &src, cv::Mat &dst);

#endif