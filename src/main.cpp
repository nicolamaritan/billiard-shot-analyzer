// Author: Nicola Maritan

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "playing_field_localizer.h"
#include "performance_measurement.h"
#include "show_cat.h"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;

    vector<Mat> found_table_masks;
    vector<Mat> ground_truth_table_masks;

    glob("*.png", filenames, true);
    for (auto filename : filenames)
    {
        if (filename.find("masks") == String::npos && filename.find("first") != String::npos)
        {
            Mat img = imread(filename);
            Mat dst;
            playing_field_localizer localizer;
            Mat found_table_mask = localizer.localize(img);
            found_table_masks.push_back(found_table_mask);
        }
    }

    //performance computation
    for (auto filename : filenames)
    {
        if (filename.find("masks") != String::npos && filename.find("first") != String::npos)
        {
            Mat img = imread(filename);
            ground_truth_table_masks.push_back(img);
        }
    }

    for (int i = 0; i < found_table_masks.size() - 1; i++)
    {
        imshow("", ground_truth_table_masks[i]);
        waitKey(0);
        imshow("", found_table_masks[i]);
        waitKey(0);
        performance_measurement measures;
        float iou_table = measures.table_segmentation_performance(ground_truth_table_masks[i], found_table_masks[i]);
    }


    return 0;
}