#ifndef PLAYING_FIELD_LOCALIZER
#define PLAYING_FIELD_LOCALIZER

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class playing_field_localizer
{
public:
    void localize(cv::Mat src, cv::Mat &dst);
private:
    void hough_approach(cv::Mat src, cv::Mat &dst);
};

#endif