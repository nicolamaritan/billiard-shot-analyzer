#ifndef PLAYING_FIELD_LOCALIZER
#define PLAYING_FIELD_LOCALIZER

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


class playing_field_localizer
{
public:
    void gamma_correction(const cv::Mat& src, cv::Mat& dst, double gamma);
    void localize(const cv::Mat& src, cv::Mat &dst);
private:
    void segmentation(const cv::Mat& src, cv::Mat& dst);
    cv::Vec3b get_board_color(const cv::Mat& src);
    double angular_coeff(const cv::Point &p1, const cv::Point &p2);
    bool is_vertical_line(const cv::Point &p1, const cv::Point &p2);
    bool are_parallel_lines(double m1, double m2);
    double intercept(const cv::Point &p1, const cv::Point &p2);
    bool is_within_image(const cv::Point &p, int rows, int cols);
    bool intersection(cv::Point o1, cv::Point p1, cv::Point o2, cv::Point p2, cv::Point &r, int rows, int cols);
    void intersections(const std::vector<std::vector<cv::Point>> &points, std::vector<cv::Point> &inters, int rows, int cols);
    double angle_between_lines(double m1, double m2);
    void draw_pool_table(std::vector<cv::Point> inters, cv::Mat& image);
    void find_lines(const cv::Mat& src);
};

#endif