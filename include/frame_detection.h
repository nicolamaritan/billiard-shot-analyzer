// Author: Nicola Maritan 2121717

#ifndef FRAME_DETECTION_H
#define FRAME_DETECTION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief Detects and draws localized playing field and balls on a frame.
 *
 * @param src The source frame to be processed.
 * @param dst The destination frame where the results will be drawn.
 */
void get_frame_detection(const cv::Mat &src, cv::Mat &dst);

#endif