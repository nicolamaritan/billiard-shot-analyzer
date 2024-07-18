#include "file_loading.h"

using namespace std;

const string MASK_NAME = "masks";
const string FIRST_NAME = "first";
const string LAST_NAME = "last";
const string BOUNDING_RECT = "bounding_boxes";

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