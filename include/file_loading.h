// Author: Nicola Maritan 2121717

#ifndef FILE_LOADING_H
#define FILE_LOADING_H

#include <string>
#include <vector>

/**
 * @brief Check if the given filename corresponds to a frame file.
 *
 * @param filename The name of the file to check.
 * @return true if the filename corresponds to a frame file, false otherwise.
 */
bool is_frame_file(const std::string &filename);

/**
 * @brief Check if the given filename corresponds to a mask frame file.
 *
 * @param filename The name of the file to check.
 * @return true if the filename corresponds to a mask frame file, false otherwise.
 */
bool is_mask_frame_file(const std::string &filename);

/**
 * @brief Check if the given filename corresponds to a bounding box file.
 *
 * @param filename The name of the file to check.
 * @return true if the filename corresponds to a bounding box file, false otherwise.
 */
bool is_bounding_box_file(const std::string &filename);

/**
 * @brief Retrieve all frame file names from the specified dataset path.
 *
 * @param dataset_path The path to the dataset directory.
 * @param frame_filenames The vector to store the retrieved frame filenames.
 */
void get_frame_files(const std::string &dataset_path, std::vector<std::string> &frame_filenames);

/**
 * @brief Retrieve all mask file names from the specified dataset path.
 *
 * @param dataset_path The path to the dataset directory.
 * @param mask_filenames The vector to store the retrieved mask filenames.
 */
void get_mask_files(const std::string &dataset_path, std::vector<std::string> &mask_filenames);

/**
 * @brief Retrieve all bounding box file names from the specified dataset path.
 *
 * @param dataset_path The path to the dataset directory.
 * @param bboxes_filenames The vector to store the retrieved bounding box filenames.
 */
void get_bounding_boxes_files(const std::string &dataset_path, std::vector<std::string> &bboxes_filenames);

#endif