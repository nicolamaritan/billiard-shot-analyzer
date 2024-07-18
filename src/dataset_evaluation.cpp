// Author: Eddie Carraro
#include "dataset_evaluation.h"
#include "performance_measurement.h"
#include "balls_localization.h"
#include "frame_segmentation.h"
#include "frame_detection.h"
#include "file_loading.h"

#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void evaluate(const string& dataset_path)
{
    const string OUTPUT_DIRECTORY = "output";
    const string PERFORMANCE_FILE = "performance.txt";

    fs::path output_directory(OUTPUT_DIRECTORY);
    fs::create_directory(output_directory);
    ofstream performance_file(output_directory /= fs::path(PERFORMANCE_FILE)); // output/performance.txt

    vector<String> filenames;
    vector<Mat> predicted_table_masks;
    vector<Mat> ground_truth_table_masks;
    vector<String> frames_filenames;
    vector<balls_localization> predicted_balls_localizations;
    vector<balls_localization> ground_truth_balls_localizations;

    // Get filenames and obtain segmentation and localization
    get_frame_files(dataset_path, filenames);
    for (auto filename : filenames)
    {
        Mat frame = imread(filename);
        Mat frame_segmentation;
        balls_localization localization;

        get_frame_segmentation(frame, frame_segmentation);
        get_balls_localization(frame, localization);

        predicted_table_masks.push_back(frame_segmentation);
        predicted_balls_localizations.push_back(localization);
    }

    get_mask_files(dataset_path, filenames);
    for (auto filename : filenames)
    {
        Mat mask = imread(filename, CV_8UC1);
        ground_truth_table_masks.push_back(mask);
    }

    get_bounding_boxes_files(dataset_path, filenames);
    for (auto filename : filenames)
    {
        balls_localization ground_truth;
        load_ground_truth_localization(filename, ground_truth);
        ground_truth_balls_localizations.push_back(ground_truth);
    }

    // Employed to output the filename in the output file
    get_frame_files(dataset_path, frames_filenames);

    for (int i = 0; i < predicted_table_masks.size(); i++)
    {
        // Evaluate and write to file
        performance_file << frames_filenames.at(i) << endl;
        performance_file << "mIoU: " << evaluate_balls_and_playing_field_segmentation(predicted_table_masks[i], ground_truth_table_masks[i]) << endl;
        performance_file << "mAP: " << evaluate_balls_localization(predicted_balls_localizations[i], ground_truth_balls_localizations[i]) << endl;
        performance_file << endl;
    }

    performance_file << "Dataset mAP: " << evaluate_balls_localization_dataset(predicted_balls_localizations, ground_truth_balls_localizations);
    performance_file.close();
}