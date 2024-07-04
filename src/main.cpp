// Author: Nicola Maritan

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "playing_field_localizer.h"
#include "balls_localizer.h"
#include "show_cat.h"

using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;
    glob("*.png", filenames, true);
    for (auto filename : filenames)
    {
        //if (filename.find("masks") == String::npos && filename.find("first") != String::npos && (filename.find("game1_clip3") != String::npos || String::npos && filename.find("game1_clip4") != String::npos))
        if (filename.find("masks") == String::npos && filename.find("first") != String::npos)
        {
            Mat img = imread(filename);

            playing_field_localizer pl_field_loc;
            pl_field_loc.localize(img);
            balls_localizer balls_loc(pl_field_loc.get_playing_field_mask(), pl_field_loc.get_playing_field_corners(), pl_field_loc.get_playing_field_hole_points());
            balls_loc.localize(img);
        }
    }

    return 0;
}