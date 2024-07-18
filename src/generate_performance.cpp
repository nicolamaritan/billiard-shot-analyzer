#include "performance_measurement.h"
#include "frame_segmentation.h"
#include "frame_detection.h"
#include "dataset_evaluation.h"

#include <iostream>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main(int argc, char** argv)
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
    
    evaluate(dataset_path);

    return 0;
}