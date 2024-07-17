#include "playing_field_segmentation.h"
#include "balls_localization.h"
#include "playing_field_localization.h"
#include "performance_measurement.h"

using namespace cv;
using namespace std;

void get_colored_segmentation(const cv::Mat& src, cv::Mat& dst, const cv::Mat& frame_segmentation, bool preserve_background);


void playing_field_segmentation(const Mat& src, Mat& dst, bool preserve_background)
{
    playing_field_localizer plf_loc;
    plf_loc.localize(src);
    balls_localizer blls_loc(plf_loc.get_localization());
    blls_loc.localize(src);

    Mat frame_segmentation;
    get_frame_segmentation(src, frame_segmentation);

    Mat colored_frame_segmentation;
    get_colored_segmentation(src, colored_frame_segmentation, frame_segmentation, preserve_background);
    dst = colored_frame_segmentation;
}

void get_colored_segmentation(const cv::Mat& src, cv::Mat& dst, const cv::Mat& frame_segmentation, bool preserve_background)
{
    dst = src.clone();
    imshow("dst", dst);

    // Define the BGR color mapping
    vector<cv::Vec3b> color_map(6);
    color_map[0] = cv::Vec3b(128, 128, 128); // Gray
    color_map[1] = cv::Vec3b(255, 255, 255); // White
    color_map[2] = cv::Vec3b(0, 0, 0);       // Black
    color_map[3] = cv::Vec3b(0, 0, 255);     // Red
    color_map[4] = cv::Vec3b(255, 0, 0);     // Blue
    color_map[5] = cv::Vec3b(0, 255, 0);     // Green

    const int MIN_COLOR = preserve_background ? 1 : 0;
    const int MAX_COLOR = 6;

    // Iterate over each pixel in the input mask
    for (int i = 0; i < frame_segmentation.rows; i++) 
    {
        for (int j = 0; j < frame_segmentation.cols; j++) 
        {
            uchar pixel_value = frame_segmentation.at<uchar>(i, j);
            if (pixel_value >= MIN_COLOR && pixel_value < MAX_COLOR) 
                dst.at<cv::Vec3b>(i, j) = color_map[pixel_value];
        }
    }
}