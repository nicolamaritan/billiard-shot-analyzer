#ifndef PLAYING_FIELD_LOCALIZATION_H
#define PLAYING_FIELD_LOCALIZATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct playing_field_localization
{
    std::vector<cv::Point> corners;
    cv::Mat mask;
    std::vector<cv::Point> hole_points;
};

typedef struct playing_field_localization playing_field_localization;

class playing_field_localizer
{
public:
    /**
     * Localize the pool table.
     *
     * @param src The input image.
     */
    void localize(const cv::Mat &src);
    playing_field_localization get_localization() { return localization; }

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
     * @brief Sorts a vector of points in a clockwise order based on their position relative to the center of the set.
     *
     * @param points Vector of points to be sorted.
     */
    void sort_points_clockwise(std::vector<cv::Point> &points);

    /**
     * @brief Estimates the locations of holes on a playing field based on the corners of the field.
     *
     * This method calculates the positions of six hole points on a playing field. It determines if the view
     * of the playing field is in perspective by analyzing the angular coefficients of the field's diagonals.
     * Depending on whether the view is perspective or not, it identifies the long and short edges of the field.
     * The method then finds the intersections of lines parallel to the short edge with the long edges to locate
     * the initial hole positions. These positions are refined to account for perspective adjustments and are
     * returned in the provided vector.
     *
     * @param hole_points A vector of Point objects where the estimated hole locations will be stored.
     */
    void estimate_holes_location(std::vector<cv::Point> &hole_points);

    playing_field_localization localization;
};

#endif