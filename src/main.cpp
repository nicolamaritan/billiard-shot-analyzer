// Author: Nicola Maritan

#include "playing_field_localization.h"
#include "balls_localization.h"
#include "minimap.h"
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

        // Load minimap
        Mat pool_table_map = imread("pool_table.png");
        Mat trajectories = pool_table_map.clone();
        minimap mini(pl_field_loc.get_localization(), balls_loc.get_localization());

        vector<Point> initial_balls_pos;
        vector<int> solids_indeces;
        vector<int> stripes_indeces;
        int black_index;
        int cue_index;
        mini.get_balls_pos(multiTracker->getObjects(), initial_balls_pos);
        mini.draw_initial_minimap(initial_balls_pos, balls_loc.get_localization(), solids_indeces, stripes_indeces, black_index, cue_index, first_frame, pool_table_map);
        imshow("initial minimap", pool_table_map);
        vector<Rect2d> old_balls_bounding_boxes = multiTracker->getObjects();

        while (cap.read(frame))
        {

            multiTracker->update(frame);

            for (const auto &object : multiTracker->getObjects())
            {
                rectangle(frame, object, Scalar(255, 0, 0), 2, 1);
            }
            vector<Point> old_balls_pos;
            mini.get_balls_pos(old_balls_bounding_boxes, old_balls_pos);

            vector<Point> current_balls_pos;
            mini.get_balls_pos(multiTracker->getObjects(), current_balls_pos);

            mini.draw_minimap(old_balls_pos, current_balls_pos, solids_indeces, stripes_indeces, black_index, cue_index, frame, trajectories, pool_table_map);
            old_balls_bounding_boxes = multiTracker->getObjects();

            imshow("MultiTracker", frame);
            imshow("Minimap", pool_table_map);
            waitKey();

            if (waitKey(1) == 'q')
            {
                break; // TODO remove
            }
        }
    }

    return 0;
}