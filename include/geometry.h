// Author: Francesco Boscolo Meneguolo 2119969

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief Computes the intersection points of given lines within specified dimensions.
 *
 * @param lines Vector of Vec3f representing the lines.
 * @param out_intersections Vector of Points where the intersection points will be stored.
 * @param rows The number of rows of the image.
 * @param cols The number of columns of the image.
 */
void intersections(const std::vector<cv::Vec3f> &lines, std::vector<cv::Point> &out_intersections, int rows, int cols);

/**
 * @brief Converts a set of lines into pairs of points for intersection calculation.
 *
 * This function takes a vector of lines, each represented by a `Vec3f`, and computes
 * two distinct points on each line. These points are stored in the output vector as pairs of `Point` objects.
 *
 * @param lines A vector of Vec3f representing the lines.
 * @param pts A vector of pairs of Point objects where the computed points for each line will be stored.
 */
void get_pairs_points_per_line(const std::vector<cv::Vec3f> &lines, std::vector<std::pair<cv::Point, cv::Point>> &pts);

/**
 * @brief Computes the intersection point of two lines if they intersect within specified dimensions.
 *
 * This function determines if two lines, represented by pairs of points, intersect within a given image
 * dimension. If the lines intersect, the intersection point is stored in the output parameter.
 *
 * @param pts_line_1 A pair of `Point` objects representing the first line.
 * @param pts_line_2 A pair of `Point` objects representing the second line.
 * @param intersection_pt A `Point` object where the intersection point will be stored if the lines intersect.
 * @param rows The number of rows of the image.
 * @param cols The number of columns of the image.
 * 
 * @return `true` if the lines intersect within the image dimensions; `false` otherwise.
 */
bool intersection(const std::pair<cv::Point, cv::Point>& pts_line_1, const std::pair<cv::Point, cv::Point>& pts_line_2, cv::Point &intersection_pt, int rows, int cols);

/**
 * @brief Computes the intersection point of two lines if they intersect within specified dimensions.
 *
 * This function determines if two lines, represented by pairs of points, intersect within a given image
 * dimension. If the lines intersect, the intersection point is stored in the output parameter.
 *
 * @param pts_line_1 A pair of `Point` objects representing the first line.
 * @param pts_line_2 A pair of `Point` objects representing the second line.
 * @param intersection_pt A `Point` object where the intersection point will be stored if the lines intersect.
 * @param rows The number of rows of the image.
 * @param cols The number of columns of the image.
 */
void intersection(const std::pair<cv::Point, cv::Point>& pts_line_1, const std::pair<cv::Point, cv::Point>& pts_line_2, cv::Point &intersection_pt);

/**
 * @brief Checks if a point lies within the bounds of an image.
 *
 * @param p A `Point` object representing the point to check.
 * @param rows The number of rows of the image.
 * @param cols The number of columns of the image.
 * @return `true` if the point is within the image dimensions; `false` otherwise.
 */
bool is_within_image(const cv::Point &p, int rows, int cols);

/**
 * @brief Computes the angular coefficient of a given line.
 *
 * This function calculates the angular coefficient of a line represented by two points.
 * If the line is vertical (i.e., the x-coordinates of both points are equal), it returns the
 * maximum possible double value to indicate an undefined slope.
 *
 * @param line A pair of `Point` objects representing the line.
 * @return The angular coefficient (slope) of the line. If the line is vertical, returns `numeric_limits<double>::max()`.
 */
double angular_coefficient(const std::pair<cv::Point, cv::Point>& line);

/**
 * @brief Computes the angle between two lines represented by pairs of points.
 *
 * This function calculates the angle between two lines given by their respective pairs of points.
 * The angle is measured in radians and ranges between 0 and π radians (0 to 180 degrees).
 *
 * @param line_1 A pair of `Point` objects representing the first line.
 * @param line_2 A pair of `Point` objects representing the second line.
 * @return The angle between the two lines in radians.
 */
double angle_between_lines(const std::pair<cv::Point, cv::Point> line_1, const std::pair<cv::Point, cv::Point> line_2);

/**
 * @brief Checks if a line is vertical based on its endpoints.
 *
 * This function determines if a line, defined by its endpoints given as a pair of `Point` objects,
 * is vertical. A line is considered vertical if both endpoints have the same x-coordinate.
 *
 * @param line A pair of `Point` objects representing the line.
 * @return `true` if the line is vertical; `false` otherwise.
 */
bool is_vertical_line(const std::pair<cv::Point, cv::Point>& line);

/**
 * @brief Checks if two lines are parallel based on their angular coefficients.
 * Two lines are considered parallel if the absolute difference between their angular coefficients
 * is within a small epsilon value.
 *
 * @param line_1 A pair of `Point` objects representing the first line.
 * @param line_2 A pair of `Point` objects representing the second line.
 * @return `true` if the lines are parallel (within a small epsilon difference in angular coefficients); `false` otherwise.
 */
bool are_parallel_lines(const std::pair<cv::Point, cv::Point>& line_1, const std::pair<cv::Point, cv::Point>& line_2);

/**
 * @brief Computes the y-intercept of a line.
 *
 * This function calculates the y-intercept of a line defined by its endpoints given as a pair
 * of `Point` objects.
 *
 * @param line A pair of `Point` objects representing the line.
 * @return The y-intercept of the line.
 */
double intercept(const std::pair<cv::Point, cv::Point>& line);

#endif