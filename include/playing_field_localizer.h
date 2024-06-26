#ifndef PLAYING_FIELD_LOCALIZER
#define PLAYING_FIELD_LOCALIZER

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class playing_field_localizer
{
public:
    void localize(const cv::Mat &src, cv::Mat &dst);

private:
    void segmentation(const cv::Mat &src, cv::Mat &dst);
    cv::Vec3b get_board_color(const cv::Mat &src, float radius);
    std::vector<cv::Vec2f> find_lines(const cv::Mat &src);
    std::vector<cv::Vec2f> refine_lines(std::vector<cv::Vec2f> &lines);
    void draw_lines(const cv::Mat &src, const std::vector<cv::Vec2f> &lines);
    void dump_similar_lines(cv::Vec2f reference_line, std::vector<cv::Vec2f> &lines, std::vector<cv::Vec2f> &similar_lines);
    void non_maxima_connected_component_suppression(const cv::Mat &src, cv::Mat &dst);
    double angular_coefficient(const cv::Point &p1, const cv::Point &p2);
    bool is_vertical_line(const cv::Point &p1, const cv::Point &p2);
    bool are_parallel_lines(double m1, double m2);
    double intercept(const cv::Point &p1, const cv::Point &p2);
    bool is_within_image(const cv::Point &p, int rows, int cols);
    bool intersection(cv::Point o1, cv::Point p1, cv::Point o2, cv::Point p2, cv::Point &r, int rows, int cols);
    void intersections(const std::vector<std::vector<cv::Point>> &points, std::vector<cv::Point> &inters, int rows, int cols);
    double angle_between_lines(double m1, double m2);
    void draw_pool_table(std::vector<cv::Point> inters, cv::Mat &image);
    void get_pairs_points_per_line(const std::vector<cv::Vec2f>& lines, std::vector<std::vector<cv::Point>> &points);
    void sort_points_clockwise(std::vector<cv::Point>& points);
};

#endif