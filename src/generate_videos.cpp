// Author: Nicola Maritan 2121717

#include "video_builder.h"

#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Wrong number of parameters. Insert the dataset location." << endl;
        return 1;
    }

    string dataset_path = static_cast<string>(argv[1]);

    video_builder builder;
    builder.build_videos(dataset_path);
}