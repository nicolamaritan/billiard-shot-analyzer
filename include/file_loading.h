#ifndef FILE_LOADING_H
#define FILE_LOADING_H

#include <string>

bool is_frame_file(std::string filename);
bool is_mask_frame_file(std::string filename);
bool is_bounding_box_file(std::string filename);

#endif