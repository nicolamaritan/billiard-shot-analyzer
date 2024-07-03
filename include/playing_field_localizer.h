#ifndef PLAYING_FIELD_LOCALIZER
#define PLAYING_FIELD_LOCALIZER

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class playing_field_localizer
{
public:
    /**
     * Localize the pool table.
     *
     * @param src The input image.
     */
    void localize(const cv::Mat &src);
    std::vector<cv::Point> get_playing_field_corners();
    cv::Mat playing_field_mask;


private:
    /**
     * @brief Perform segmentation of the image based on color. One of the clusters should
     * contain the whole table, surrounded by different clusters.
     *
     * @param src The input image to segment.
     * @param dst The segmented image.
     */
    void segmentation(const cv::Mat &src, cv::Mat &dst);

    /**
     * @brief Return the estimated board color.
     *
     * It computes the color of the board by considering a circle of a given radius around
     * the center of the image and picking the median value.
     *
     * @param src Input image containing the board.
     * @param radius Radius from the image center in which to compute the board color.
     * @return the computed color of the board.
     */
    cv::Vec3b get_board_color(const cv::Mat &src, float radius);

    /**
     * @brief Finds lines in the input edge-detected image using the Hough Transform.
     *
     * @param edges Input edge-detected binary image.
     * @param lines Output vector of lines, each represented by a Vec3f (rho, theta, number of votes).
     */
    void find_lines(const cv::Mat &src, std::vector<cv::Vec3f> &lines);

    /**
     * @brief Refines a vector of lines by eliminating similar lines. Similar lines are condensed
     *        into a single line by computing their mean values.
     *
     * @param lines Input vector of lines, each represented by a Vec3f (rho, theta, number of votes).
     * @param refined_lines Output vector of refined lines.
     */
    void refine_lines(const std::vector<cv::Vec3f> &lines, std::vector<cv::Vec3f> &refined_lines);

    /**
     * @brief Draws lines on the input image and displays it.
     *
     * @param src Input image on which lines will be drawn.
     * @param lines Vector of lines to be drawn, each represented by a Vec3f (rho, theta, line_id).
     */
    void draw_lines(const cv::Mat &src, const std::vector<cv::Vec3f> &lines);

    /**
     * @brief Finds and removes lines similar to a reference line from a vector of lines.
     *
     * @param reference_line The line to which other lines will be compared.
     * @param lines Input vector of lines, from which similar lines will be removed.
     * @param similar_lines Output vector of lines similar to the reference line.
     * @param rho_threshold Threshold for the difference in rho to consider lines similar.
     * @param theta_threshold Threshold for the difference in theta to consider lines similar.
     */
    void dump_similar_lines(const cv::Vec3f &reference_line, std::vector<cv::Vec3f> &lines, std::vector<cv::Vec3f> &similar_lines, float rho_threshold, float theta_threshold);

    /**
     * @brief Performs non-maxima suppression on the connected components of the input image,
     *        keeping only the largest component.
     *
     * @param src Input binary image.
     * @param dst Output image where only the largest connected component is retained.
     */
    void non_maxima_connected_component_suppression(const cv::Mat &src, cv::Mat &dst);

    /**
     * @brief Checks if a given point is within the bounds of an image.
     *
     * @param pt The point to check.
     * @param rows The number of rows in the image.
     * @param cols The number of columns in the image.
     * @return true If the point is within the image bounds.
     * @return false If the point is outside the image bounds.
     */
    bool is_within_image(const cv::Point &pt, int rows, int cols);

    /**
     * @brief Finds the intersection point of two lines if it lies within the image bounds.
     *
     * @param pts_line_1 Pair of points defining the first line.
     * @param pts_line_2 Pair of points defining the second line.
     * @param intersection_pt The output point of intersection.
     * @param rows The number of rows in the image.
     * @param cols The number of columns in the image.
     * @return true If the intersection point is within the image bounds.
     * @return false If the lines do not intersect within the image bounds or are parallel.
     */
    bool intersection(std::pair<cv::Point, cv::Point> pts_line_1, std::pair<cv::Point, cv::Point> pts_line_2, cv::Point &intersection_pt, int rows, int cols);

    /**
     * @brief Finds all intersections between pairs of lines and stores them if they lie within the image bounds.
     *
     * @param lines Vector of lines, each represented by a Vec3f (rho, theta, number of votes).
     * @param out_intersections Vector of intersection points within the image bounds.
     * @param rows The number of rows in the image.
     * @param cols The number of columns in the image.
     */
    void intersections(const std::vector<cv::Vec3f> &lines, std::vector<cv::Point> &out_intersections, int rows, int cols);

    /**
     * @brief Converts a set of lines represented by (rho, theta) into pairs of points.
     *
     * @param lines Vector of lines, each represented by a Vec3f (rho, theta, number of votes).
     * @param pts Output vector of pairs of points, each pair representing a line.
     */
    void get_pairs_points_per_line(const std::vector<cv::Vec3f> &lines, std::vector<std::pair<cv::Point, cv::Point>> &pts);

    /**
     * @brief Sorts a vector of points in a clockwise order based on their position relative to the center of the set.
     *
     * @param points Vector of points to be sorted.
     */
    void sort_points_clockwise(std::vector<cv::Point> &points);

    double angle_between_lines(double m1, double m2);
    void draw_pool_table(std::vector<cv::Point> inters, cv::Mat &image);
    double angular_coefficient(const cv::Point &pt1, const cv::Point &pt2);
    bool is_vertical_line(const cv::Point &pt1, const cv::Point &pt2);
    bool are_parallel_lines(double m1, double m2);
    double intercept(const cv::Point &pt1, const cv::Point &pt2);

    std::vector<cv::Point> playing_field_corners;
};

#endif