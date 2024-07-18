#include "file_loading.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

const string MASK_NAME = "masks";
const string FIRST_NAME = "first";
const string LAST_NAME = "last";
const string BOUNDING_RECT = "bounding_boxes";
const string PNG_EXTENSION = "*.png";
const string TXT_EXTENSION = "*.txt";

bool is_frame_file(string filename)
{
    return filename.find("masks") == string::npos && (filename.find("first") != string::npos || filename.find("last") != string::npos);
}

bool is_mask_frame_file(string filename)
{
    return filename.find("masks") != string::npos && (filename.find("first") != string::npos || filename.find("last") != string::npos);
}

bool is_bounding_box_file(string filename)
{
    return filename.find("bounding_boxes") != string::npos && (filename.find("first") != string::npos || filename.find("last") != string::npos);
}

void get_frame_files(std::string dataset_path, std::vector<std::string> &frame_filenames)
{
    frame_filenames.clear();
    vector<string> filenames;
    glob(dataset_path + PNG_EXTENSION, filenames, true);

    for (auto filename : filenames)
    {
        if (is_frame_file(filename))
        {
            frame_filenames.push_back(filename);
        }
    }
}

void get_mask_files(std::string dataset_path, std::vector<std::string> &mask_filenames)
{
    mask_filenames.clear();
    vector<string> filenames;
    glob(dataset_path + PNG_EXTENSION, filenames, true);

    for (auto filename : filenames)
    {
        if (is_mask_frame_file(filename))
        {
            mask_filenames.push_back(filename);
        }
    }
}

void get_bounding_boxes_files(std::string dataset_path, std::vector<std::string> &bboxes_filenames)
{
    bboxes_filenames.clear();
    vector<string> filenames;
    glob(dataset_path + TXT_EXTENSION, filenames, true);

    for (auto filename : filenames)
    {
        if (is_bounding_box_file(filename))
        {
            bboxes_filenames.push_back(filename);
        }
    }
}