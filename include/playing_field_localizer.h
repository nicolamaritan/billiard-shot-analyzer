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
    std::vector<cv::Vec2f> find_lines(const cv::Mat& src);
    std::vector<cv::Vec2f> refine_lines(std::vector<cv::Vec2f>& lines);
    void draw_lines(const cv::Mat &src, const std::vector<cv::Vec2f>& lines);
    void dump_similar_lines(cv::Vec2f reference_line, std::vector<cv::Vec2f>& lines, std::vector<cv::Vec2f>& similar_lines);
};

#endif