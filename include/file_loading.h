// Author: Nicola Maritan 2121717

#ifndef FILE_LOADING_H
#define FILE_LOADING_H

#include <string>
#include <vector>

/**
 * @brief Check if the given filename corresponds to a frame file.
 * 
 * Frame files contain "first" or "last" in their names but do not contain "masks".
 * 
 * @param filename The name of the file to check.
 * @return true if the filename corresponds to a frame file, false otherwise.
 */
bool is_frame_file(std::string filename);

/**
 * @brief Check if the given filename corresponds to a mask frame file.
 * 
 * Mask frame files contain "masks" and either "first" or "last" in their names.
 * 
 * @param filename The name of the file to check.
 * @return true if the filename corresponds to a mask frame file, false otherwise.
 */
bool is_mask_frame_file(std::string filename);

/**
 * @brief Check if the given filename corresponds to a bounding box file.
 * 
 * Bounding box files contain "bounding_boxes" and either "first" or "last" in their names.
 * 
 * @param filename The name of the file to check.
 * @return true if the filename corresponds to a bounding box file, false otherwise.
 */
bool is_bounding_box_file(std::string filename);

/**
 * @brief Retrieve all frame file names from the specified dataset path.
 * 
 * This function clears the provided vector and populates it with filenames 
 * from the dataset path that match the frame file criteria.
 * 
 * @param dataset_path The path to the dataset directory.
 * @param frame_filenames The vector to store the retrieved frame filenames.
 */
void get_frame_files(std::string dataset_path, std::vector<std::string> &frame_filenames);

/**
 * @brief Retrieve all mask file names from the specified dataset path.
 * 
 * This function clears the provided vector and populates it with filenames 
 * from the dataset path that match the mask file criteria.
 * 
 * @param dataset_path The path to the dataset directory.
 * @param mask_filenames The vector to store the retrieved mask filenames.
 */
void get_mask_files(std::string dataset_path, std::vector<std::string> &mask_filenames);

/**
 * @brief Retrieve all bounding box file names from the specified dataset path.
 * 
 * This function clears the provided vector and populates it with filenames 
 * from the dataset path that match the bounding box file criteria.
 * 
 * @param dataset_path The path to the dataset directory.
 * @param bboxes_filenames The vector to store the retrieved bounding box filenames.
 */
void get_bounding_boxes_files(std::string dataset_path, std::vector<std::string> &bboxes_filenames);

#endif