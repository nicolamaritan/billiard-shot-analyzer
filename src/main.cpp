// Author: Nicola Maritan

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "playing_field_localizer.h"
#include "balls_localizer.h"

using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;
    glob("*.png", filenames, true);
    for (auto filename : filenames)
    {
        if (filename.find("masks") == String::npos && filename.find("first") != String::npos)
        {
            Mat img = imread(filename);
            Mat dst1, dst2;
            playing_field_localizer localizer;
            vector<Point> corners;
            corners = localizer.localize(img, dst1);
            balls_localizer balls_loc;
            balls_loc.localize_balls(img, dst2, localizer.temp_edges, corners);

        }
    }

    return 0;
}