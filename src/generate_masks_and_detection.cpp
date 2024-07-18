#include "balls_localization.h"
#include "frame_segmentation.h"
#include "frame_detection.h"

#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    string dataset_path = "./dataset/";

    fs::path output_directory("output");
    fs::path masks_and_detection_directory("masks_and_detection");
    output_directory /= masks_and_detection_directory;
    fs::create_directories(output_directory);   // ./output/masks_and_detection

    vector<String> filenames;
    glob(dataset_path + "*.png", filenames, true);
    for (auto filename : filenames)
    {
        if (filename.find("masks") == String::npos && (filename.find("first") != String::npos || filename.find("last") != String::npos))
        {
            Mat frame = imread(filename);
            Mat frame_segmentation;
            Mat frame_segmentation_background_preserved;
            Mat frame_detection;

            fs::path file_path = fs::path(filename);
            fs::path clip_game_directory;
            clip_game_directory /= output_directory;
            clip_game_directory /= file_path.parent_path().parent_path().filename(); // Create directory of type "game{a}_clip{b}, etc"  
            fs::create_directory(clip_game_directory);

            clip_game_directory /= fs::path(filename).filename();   // output/masks_and_detection/game{a}_clip{b}

            const string SEGMENTATION_FILENAME = "_segmentation"; 
            const string SEGMENTATION_BACKGROUND_PRESERVED_FILENAME = "_segmentation_background_preserved"; 
            const string DETECTION_FILENAME = "_detection"; 

            get_colored_frame_segmentation(frame, frame_segmentation, false);
            get_colored_frame_segmentation(frame, frame_segmentation_background_preserved, true);
            get_frame_detection(frame, frame_detection);

            // Save files
            imwrite(clip_game_directory.string() + SEGMENTATION_FILENAME, frame_segmentation);
            imwrite(clip_game_directory.string() + SEGMENTATION_BACKGROUND_PRESERVED_FILENAME, frame_segmentation_background_preserved);
            imwrite(clip_game_directory.string() + DETECTION_FILENAME, frame_detection);
        }
    }
}