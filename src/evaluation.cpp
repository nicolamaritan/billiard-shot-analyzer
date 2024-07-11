#include "playing_field_localizer.h"
#include "balls_localizer.h"

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;
    glob("frame_*.png", filenames, true);
    for (auto filename : filenames)
    {
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