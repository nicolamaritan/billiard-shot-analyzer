// Author: Nicola Maritan

#include "playing_field_localizer.h"
#include "balls_localizer.h"
#include "show_cat.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;
    glob("*.mp4", filenames, true);
    for (String filename : filenames)
    {
        VideoCapture cap(filename); // open the default camera
        if (!cap.isOpened())        // check if we succeeded
            return -1;

        Mat first_frame, frame;
        cap.read(first_frame);
        playing_field_localizer pl_field_loc;
        pl_field_loc.localize(first_frame);

        balls_localizer balls_loc(pl_field_loc.get_localization());
        balls_loc.localize(first_frame);

        // Create a MultiTracker object
        Ptr<legacy::MultiTracker> multiTracker = legacy::MultiTracker::create();

        // Initialize the trackers for each ROI
        for (const auto &roi : balls_loc.get_bounding_boxes())
        {
            multiTracker->add(legacy::TrackerCSRT::create(), first_frame, roi);
        }

        while (cap.read(frame))
        {
            multiTracker->update(frame);

            for (const auto &object : multiTracker->getObjects())
            {
                rectangle(frame, object, Scalar(255, 0, 0), 2, 1);
            }

            imshow("MultiTracker", frame);

            if (waitKey(1) == 'q')
            {
                break;
            }
        }
    }

    return 0;
}