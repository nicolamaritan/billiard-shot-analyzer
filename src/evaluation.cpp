#include "performance_measurement.h"

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    vector<String> filenames;

    vector<Mat> found_table_masks;
    vector<Mat> ground_truth_table_masks;
    vector<String> frames_filenames;

    glob("*.png", filenames, true);
    for (auto filename : filenames)
    {
        if (filename.find("masks") == String::npos && (filename.find("first") != String::npos || filename.find("last") != String::npos))
        {
            Mat img = imread(filename);
            Mat img_segmentation;

            frame_segmentation(img, img_segmentation);
            found_table_masks.push_back(img_segmentation);
            frames_filenames.push_back(filename);
        }
    }

    //performance computation
    for (auto filename : filenames)
    {
        if (filename.find("masks") != String::npos && (filename.find("first") != String::npos || filename.find("last") != String::npos))
        {
            Mat img = imread(filename, CV_8UC1);
            ground_truth_table_masks.push_back(img);
        }
    }

    cout << "import and segmentation done" << endl;
    for (int i = 0; i < found_table_masks.size(); i++)
    {
        cout << frames_filenames.at(i) << endl;
        evaluate_balls_and_playing_field_segmentation(ground_truth_table_masks[i], found_table_masks[i]);
    }


    return 0;
}