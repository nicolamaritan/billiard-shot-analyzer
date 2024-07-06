// Author: Nicola Maritan

#include "playing_field_localizer.h"
#include "balls_localizer.h"
#include "show_cat.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;
    glob("*.mp4", filenames, true);
    for (auto filename : filenames)
    {
        int n_frames = 3;
        //if (filename.find(".mp4") != String::npos && filename.find("clip4") != String::npos)
        if (filename.find(".mp4") != String::npos )
        {
            VideoCapture cap(filename); // open the default camera
            if (!cap.isOpened())        // check if we succeeded
                return -1;

            Mat first_frame;
            cap.read(first_frame);
            playing_field_localizer pl_field_loc;
            pl_field_loc.localize(first_frame);
            for (;;)
            {
                Mat frame;
                cap.read(frame);

                if (frame.empty())
                {
                    cerr << "ERROR! blank frame grabbed\n";
                    break;
                }

                // Mat img = imread(filename);

                balls_localizer balls_loc(pl_field_loc.get_playing_field_mask(), pl_field_loc.get_playing_field_corners(), pl_field_loc.get_playing_field_hole_points());
                balls_loc.localize(frame);

                // show live and wait for a key with timeout long enough to show images
                // imshow("Live", frame);
                waitKey();
                if (--n_frames == 0)
                    break;
                //break;
                // if (waitKey(5) >= 0)
                //     break;
            }
        }
    }

    return 0;
}