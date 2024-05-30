// Author: Nicola Maritan

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "show_cat.h"

using namespace std;
using namespace cv;

const string SAMPLE_IMAGE_WINDOW_NAME = "Sample image.";

void show_cat()
{
    Mat img = imread("images/cat.jpg");
    namedWindow(SAMPLE_IMAGE_WINDOW_NAME);
    imshow(SAMPLE_IMAGE_WINDOW_NAME, img);
    waitKey(0);
}