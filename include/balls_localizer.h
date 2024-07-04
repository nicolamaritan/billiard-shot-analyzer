#ifndef BALLS_LOCALIZER
#define BALLS_LOCALIZER

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class balls_localizer
{
public:
    void localize_balls(const cv::Mat &src, cv::Mat &mask, const cv::Mat temp_edges, std::vector<cv::Point> corners);

private:
    void create_table_mask(const cv::Mat &src, cv::Mat &dst, std::vector<cv::Point> corners);
    bool is_same_color(const cv::Vec3b& color1, const cv::Vec3b& color2);
    void process_table(cv::Mat &image, const cv::Vec3b& color, int row, int col);
    void apply_mask(cv::Mat &image, const cv::Mat &mask);
    void localize_red_balls(const cv::Mat &src, cv::Mat &dst);

    

};

#endif