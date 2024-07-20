// Author: Nicola Maritan 2121717

#include "balls_localization.h"
#include "frame_segmentation.h"
#include "frame_detection.h"
#include "file_loading.h"

#include <fstream>
#include <filesystem>
#include <iostream>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Wrong number of parameters. Insert the dataset location." << endl;
        return 1;
    }

    string dataset_path = static_cast<string>(argv[1]);
    if (!fs::is_directory(dataset_path))
    {
        cerr << "Dataset directory not found." << endl;
        return 1;
    }

    // Add OS separator if not inserted
    if (dataset_path.back() != fs::path::preferred_separator)
        dataset_path = dataset_path + fs::path::preferred_separator;

    fs::path output_directory("output");
    fs::path masks_and_detection_directory("masks_and_detection");
    output_directory /= masks_and_detection_directory;
    fs::create_directories(output_directory); // .{dataset}/output/masks_and_detection

    vector<String> filenames;
    get_frame_files(dataset_path, filenames);

    for (auto filename : filenames)
    {
        Mat frame = imread(filename);
        Mat frame_segmentation;
        Mat frame_segmentation_background_preserved;
        Mat frame_detection;

        fs::path file_path = fs::path(filename);
        fs::path clip_game_directory;
        clip_game_directory /= output_directory;
        clip_game_directory /= file_path.parent_path().parent_path().filename(); // Create directory of type "game{a}_clip{b}"
        fs::create_directory(clip_game_directory);

        clip_game_directory /= fs::path(filename).filename(); // {dataset}/output/masks_and_detection/game{a}_clip{b}

        const string SEGMENTATION_FILENAME = "_segmentation";
        const string SEGMENTATION_BACKGROUND_PRESERVED_FILENAME = "_segmentation_background_preserved";
        const string DETECTION_FILENAME = "_detection";

        cout << "Generating masks and detections in " << clip_game_directory.string() << "..." << endl;

        // Compute output images
        get_colored_frame_segmentation(frame, frame_segmentation, false);
        get_colored_frame_segmentation(frame, frame_segmentation_background_preserved, true);
        get_frame_detection(frame, frame_detection);

        // Save files
        imwrite(clip_game_directory.string() + SEGMENTATION_FILENAME, frame_segmentation);
        imwrite(clip_game_directory.string() + SEGMENTATION_BACKGROUND_PRESERVED_FILENAME, frame_segmentation_background_preserved);
        imwrite(clip_game_directory.string() + DETECTION_FILENAME, frame_detection);
        cout << "Generated masks and detections in " << clip_game_directory.string() << endl;
    }

    return 0;
}