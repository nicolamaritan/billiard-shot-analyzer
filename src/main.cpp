// Author: Nicola Maritan

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "playing_field_localizer.h"
#include "show_cat.h"

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
            Mat dst;
            playing_field_localizer localizer;
            localizer.localize(img, dst);
        }
    }

    return 0;
}