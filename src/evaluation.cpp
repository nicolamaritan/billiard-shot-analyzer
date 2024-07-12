#include "playing_field_localization.h"
#include "balls_localization.h"

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;
    glob("frame_*.png", filenames, true);
    for (auto filename : filenames)
    {
        // Avoid ground-truth masks
        if (filename.find("frames") != String::npos)
        {
            Mat image = imread(filename);

            playing_field_localizer localizer_1;
            localizer_1.localize(image);
            balls_localizer localizer_2(localizer_1.get_localization());
            localizer_2.localize(image);

            waitKey();
        }
    }

    return 0;
}