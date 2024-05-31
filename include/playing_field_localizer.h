#ifndef PLAYING_FIELD_LOCALIZER
#define PLAYING_FIELD_LOCALIZER

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class playing_field_localizer
{
public:
    void localize(const cv::Mat& src, cv::Mat &dst);
private:
    void segmentation(const cv::Mat& src, cv::Mat& dst);
    cv::Vec3b get_board_color(const cv::Mat& src);
    void find_lines(const cv::Mat& src);
};

#endif