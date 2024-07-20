// Author: Nicola Maritan 2121717

#ifndef PERFORMANCE_MEASUREMENT_H
#define PERFORMANCE_MEASUREMENT_H

#include "balls_localization.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * @enum match_type
 * @brief Enum representing the type of match result.
 * 
 * @var false_positive Represents a false positive match.
 * @var true_positive Represents a true positive match.
 */
enum match_type
{
    false_positive,
    true_positive
};

/**
 * @struct match
 * @brief Struct representing the match result and confidence.
 * 
 * @var type The type of match (true_positive or false_positive).
 * @var confidence The confidence score of the match.
 */
struct match
{
    match_type type;
    float confidence;
};
typedef struct match match;

/**
 * @brief Evaluate the performance of ball and playing field segmentation on a dataset.
 * 
 * @param predicted_masks A vector of predicted masks.
 * @param ground_truth_masks A vector of ground truth masks.
 * @return The mean IoU value.
 */
float evaluate_balls_localization_dataset(const std::vector<balls_localization> &predicted_localizations, const std::vector<balls_localization> &ground_truth_localizations);

/**
 * @brief Evaluate the performance of ball localization on a dataset.
 * 
 * @param predicted_localizations A vector of predicted ball localizations.
 * @param ground_truth_localizations A vector of ground truth ball localizations.
 * @return The mean average precision value.
 */
float evaluate_balls_and_playing_field_segmentation_dataset(const std::vector<cv::Mat> &predicted_masks, const std::vector<cv::Mat> &ground_truth_masks);

/**
 * @brief Evaluate the performance of ball localization.
 * 
 * @param predicted The predicted balls localization.
 * @param ground_truth The ground truth balls localization.
 * @return The mean average precision value.
 */
float evaluate_balls_localization(const balls_localization &predicted_localization, const balls_localization &ground_truth_localization);

/**
 * @brief Evaluate the performance of ball and playing field segmentation.
 * 
 * @param found_mask The predicted mask.
 * @param ground_truth_mask The ground truth mask.
 * @return The mean IoU value.
 */
float evaluate_balls_and_playing_field_segmentation(const cv::Mat &predicted_mask, const cv::Mat &ground_truth_mask);

/**
 * @brief Perform ball localization on an image and store the results.
 * 
 * @param src The source image.
 * @param localization The output ball localizations.
 */
void get_balls_localization(const cv::Mat &src, balls_localization &localization);

/**
 * @brief Load ground truth ball localization from a file.
 * 
 * @param filename The filename containing the ground truth data.
 * @param ground_truth_localization The output ground truth ball localizations.
 */
void load_ground_truth_localization(const std::string &filename, balls_localization &ground_truth_localization);

#endif