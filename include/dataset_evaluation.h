// Author: Eddie Carraro
#ifndef EVALUATE_DATASET_H
#define EVALUATE_DATASET_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief Processes images and associated data from a specified directory, evaluates the performance of table segmentation and ball localization, and writes the results (the mean Intersection over Union and the mean Average Precision) to a text file.
 *
 * @param dataset_path A string representing the directory path containing the images and ground truth files.
 */
void evaluate(const std::string& dataset_path);

#endif