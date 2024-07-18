#include "dataset_evaluation.h"
#include "performance_measurement.h"
#include "balls_localization.h"
#include "frame_segmentation.h"
#include "frame_detection.h"

#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;


void evaluate(std::string path)
{
    vector<String> filenames;
    vector<Mat> predicted_table_masks;
    vector<Mat> ground_truth_table_masks;
    vector<String> frames_filenames;
    vector<balls_localization> predicted_balls_localizations;
    vector<balls_localization> ground_truth_balls_localizations;

    namespace fs = std::filesystem;
    fs::create_directories("./output");
    ofstream performance_file("./output/performance.txt");

    glob(path + "*.png", filenames, true);

    for (auto filename : filenames)
    {
        if (filename.find("masks") == String::npos && (filename.find("first") != String::npos || filename.find("last") != String::npos))
        {
            Mat img = imread(filename);
            Mat img_segmentation;
            balls_localization localization;

            get_frame_segmentation(img, img_segmentation);
            get_balls_localization(img, localization);

            predicted_table_masks.push_back(img_segmentation);
            frames_filenames.push_back(filename);
            predicted_balls_localizations.push_back(localization);
        }
    }

    for (auto filename : filenames)
    {
        if (filename.find("masks") != String::npos && (filename.find("first") != String::npos || filename.find("last") != String::npos))
        {
            Mat img = imread(filename, CV_8UC1);
            ground_truth_table_masks.push_back(img);
        }
    }

    // Bounding box load
    vector<String> txt_filenames;
    glob(path + "*.txt", txt_filenames, true);
    for (auto txt_filename : txt_filenames)
    {
        if (txt_filename.find("bounding_boxes") != String::npos && (txt_filename.find("first") != String::npos || txt_filename.find("last") != String::npos))
        {
            balls_localization ground_truth;
            load_ground_truth_localization(txt_filename, ground_truth);
            ground_truth_balls_localizations.push_back(ground_truth);
        }
    }

    for (int i = 0; i < predicted_table_masks.size(); i++)
    {
        performance_file << frames_filenames.at(i) << endl;
        performance_file << "mIoU: " << evaluate_balls_and_playing_field_segmentation(predicted_table_masks[i], ground_truth_table_masks[i]) << endl;
        performance_file << "mAP: " << evaluate_balls_localization(predicted_balls_localizations[i], ground_truth_balls_localizations[i]) << endl;
        performance_file << endl;
    }

    performance_file << "Dataset mAP: " << evaluate_balls_localization_dataset(predicted_balls_localizations, ground_truth_balls_localizations);

    performance_file.close();
}