// Author: Nicola Maritan 2121717

#include "video_builder.h"

#include <iostream>

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Wrong number of parameters. Insert the dataset location." << endl;
        return 1;
    }

    string dataset_path = static_cast<string>(argv[1]);

    // Add OS separator if not inserted
    if (dataset_path.back() != fs::path::preferred_separator)
        dataset_path = dataset_path + fs::path::preferred_separator;

    video_builder builder;

    try
    {
        builder.build_videos(dataset_path);
    }
    catch (const exception &e)
    {
        cerr << e.what() << endl;
        cerr << "Terminating the program" << endl;
        return 1;
    }

    return 0;
}