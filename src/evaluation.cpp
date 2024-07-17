#include "performance_measurement.h"
#include "frame_segmentation.h"
#include "frame_detection.h"
#include "dataset_evaluation.h"

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    evaluate("./dataset/");

    return 0;
}