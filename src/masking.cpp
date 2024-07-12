#include "masking.h"

using namespace std;
using namespace cv;

void mask_hsv(const cv::Mat &src, cv::Mat &dst, const cv::Mat &mask)
{
    Mat mask_hsv, mask_bgr;
    cvtColor(mask, mask_bgr, COLOR_GRAY2BGR);
    cvtColor(mask_bgr, mask_hsv, COLOR_BGR2HSV);
    bitwise_and(src, mask_hsv, dst);
}

void mask_bgr(const cv::Mat &src, cv::Mat &dst, const cv::Mat &mask)
{
    Mat mask_bgr;
    cvtColor(mask, mask_bgr, COLOR_GRAY2BGR);
    bitwise_and(src, mask_bgr, dst);
}