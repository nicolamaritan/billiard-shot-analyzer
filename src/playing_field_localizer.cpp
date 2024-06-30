#include <iostream>
#include <cmath>
#include <map>
#include "playing_field_localizer.h"

using namespace cv;
using namespace std;

void playing_field_localizer::segmentation(const Mat &src, Mat &dst)
{
    // HSV allows to separate brightness from other color characteristics, therefore
    // it is employed for kmeans clustering.
    cvtColor(src, dst, COLOR_BGR2HSV);

    // Apply uniform Value (of HSV) for the whole image, to handle different brightnesses
    const int VALUE_UNIFORM = 128;
    vector<Mat> hsv_channels;
    split(dst, hsv_channels);
    hsv_channels[2].setTo(VALUE_UNIFORM);
    merge(hsv_channels, dst);

    // data contains dst data (init with src data) used for kmeans clustering (therefore employs 32-bit float values)
    Mat data;
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // Image segmentation is performed via kmeans on the hsv img
    Mat labels, centers;
    const int NUMBER_OF_CENTERS = 3;
    const int KMEANS_MAX_COUNT = 10;
    const int KMEANS_EPSILON = 1.0;
    const int KMEANS_ATTEMPTS = 3;
    kmeans(data, NUMBER_OF_CENTERS, labels, TermCriteria(TermCriteria::MAX_ITER, KMEANS_MAX_COUNT, KMEANS_EPSILON), KMEANS_ATTEMPTS, KMEANS_PP_CENTERS, centers);

    // Reshape both to a single row of Vec3f pixels
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    // Replace pixel values with their centroids value
    for (int i = 0; i < data.rows; i++)
    {
        int center_id = labels.at<int>(i);
        data.at<Vec3f>(i) = centers.at<Vec3f>(center_id);
    }

    dst = data.reshape(3, dst.rows);
    dst.convertTo(dst, CV_8U);
}

Vec3b playing_field_localizer::get_board_color(const Mat &src, float radius)
{
    int center_cols = src.cols / 2;
    int center_rows = src.rows / 2;
    vector<Vec3b> pixel_values;

    // Collect all pixel values in a radius 'radius' around the image center.
    for (int row = -radius; row <= radius; ++row)
    {
        for (int col = -radius; col <= radius; ++col)
        {
            if (col * col + row * row <= radius * radius)
            {
                int current_row = center_rows + row;
                int current_col = center_cols + col;

                if (current_row >= 0 && current_row < src.rows && current_col >= 0 && current_col < src.cols)
                {
                    pixel_values.push_back(src.at<Vec3b>(current_row, current_col));
                }
            }
        }
    }

    // Return black if no pixel_values are collected
    if (pixel_values.empty())
    {
        return Vec3b(0, 0, 0);
    }

    // Sort by norm. In a grayscale context, we would have just considered the pixel intensity.
    // However, now we have 3 components. So we sort the pixel values triplets by their norm.
    sort(pixel_values.begin(), pixel_values.end(), [](const Vec3b &a, const Vec3b &b)
         { return norm(a) < norm(b); });

    return pixel_values[pixel_values.size() / 2];
}

void playing_field_localizer::find_lines(const Mat &edges, vector<Vec3f> &lines)
{
    const float RHO_RESOLUTION = 1.5;   // In pixels.
    const float THETA_RESOLUTION = 1.8; // In radians.
    const int THRESHOLD = 120;

    Mat cdst;
    cvtColor(edges, cdst, COLOR_GRAY2BGR);
    HoughLines(edges, lines, RHO_RESOLUTION, THETA_RESOLUTION * CV_PI / 180, THRESHOLD, 0, 0);

    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
    }

    imshow("", cdst);
    waitKey();
}

/**
 * Localize the pool table.
 *
 * @param src The input image.
 * @param dst The destination image containing the localized table.
 */
void playing_field_localizer::localize(const Mat &src, Mat &dst)
{
    const int FILTER_SIZE = 3;
    const int FILTER_SIGMA = 20;
    GaussianBlur(src.clone(), src, Size(FILTER_SIZE, FILTER_SIZE), FILTER_SIGMA, FILTER_SIGMA);

    Mat segmented, labels;
    segmentation(src, segmented);

    imshow("", segmented);
    waitKey(0);

    const int RADIUS = 30;
    Vec3b board_color = get_board_color(segmented, RADIUS);

    Mat mask;
    inRange(segmented, board_color, board_color, mask);
    segmented.setTo(Scalar(0, 0, 0), mask);

    imshow("", mask);
    waitKey(0);

    non_maxima_connected_component_suppression(mask.clone(), mask);
    imshow("", mask);
    waitKey(0);

    const int THRESHOLD_1_CANNY = 50;
    const int THRESHOLD_2_CANNY = 150;
    Mat edges;
    Canny(mask, edges, THRESHOLD_1_CANNY, THRESHOLD_2_CANNY);
    imshow("", edges);
    waitKey(0);

    vector<Vec3f> lines, refined_lines;
    find_lines(edges, lines);
    refine_lines(lines, refined_lines);

    draw_lines(edges, refined_lines);

    vector<vector<Point>> points_refined_line;
    get_pairs_points_per_line(refined_lines, points_refined_line);

    vector<Point> refined_lines_intersections;
    Mat table = src.clone();
    intersections(points_refined_line, refined_lines_intersections, table.rows, table.cols);
    draw_pool_table(refined_lines_intersections, table);
    imshow("", table);
    waitKey(0);

    sort_points_clockwise(refined_lines_intersections);

    fillConvexPoly(table, refined_lines_intersections, Scalar(0, 0, 255));
    imshow("", table);
    waitKey(0);
}

/**
 * Compute a vector of refined line by eliminating similar lines. Similar lines are condensed
 * to a single line by computing their mean values.
 */
void playing_field_localizer::refine_lines(const vector<Vec3f> &lines, vector<Vec3f> &refined_lines)
{
    vector<Vec3f> lines_copy = lines;

    while (!lines_copy.empty())
    {
        Vec3f reference_line = lines_copy.back();
        lines_copy.pop_back();
        vector<Vec3f> similar_lines;

        dump_similar_lines(reference_line, lines_copy, similar_lines);

        // Compute a new mean line with the dumped ones
        Vec3f mean_line;
        int total_votes = 0;
        for (auto similar_line : similar_lines)
        {
            total_votes += similar_line[2];
            mean_line += similar_line * similar_line[2];
        }
        mean_line[0] /= total_votes;
        mean_line[1] /= total_votes;
        refined_lines.push_back(mean_line);
    }
}

void playing_field_localizer::draw_lines(const Mat &src, const vector<Vec3f> &lines)
{
    Mat src_bgr;
    cvtColor(src, src_bgr, COLOR_GRAY2BGR);

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(src_bgr, pt1, pt2, Scalar(0, 255, 0), 1, LINE_AA);
    }

    imshow("", src_bgr);
    waitKey();
}

void playing_field_localizer::dump_similar_lines(const Vec3f& reference_line, vector<Vec3f> &lines, vector<Vec3f> &similar_lines)
{
    const float RHO_THRESHOLD = 25;
    const float THETA_THRESHOLD = 0.2;
    similar_lines.push_back(reference_line);

    // Insert into similar_lines all the similar lines and removes them from lines.
    int i = 0;
    while (i < lines.size())
    {
        Vec3f line = lines.at(i);
        if (abs(line[0] - reference_line[0]) < RHO_THRESHOLD && abs(line[1] - reference_line[1]) < THETA_THRESHOLD)
        {
            similar_lines.push_back(line);
            lines.erase(lines.begin() + i);
        }
        else
        {
            i++;
        }
    }
}

void playing_field_localizer::non_maxima_connected_component_suppression(const Mat &src, Mat &dst)
{
    src.copyTo(dst);
    Mat connected_components_labels, stats, centroids;
    connectedComponentsWithStats(src, connected_components_labels, stats, centroids);

    const int AREA_STAT_ID = 4;
    int max_label_component = 1;
    int max_area = 0;

    // Find component with greatest area
    for (int i = 1; i < stats.rows; i++)
    {
        int component_area = stats.at<int>(i, AREA_STAT_ID);
        if (component_area > max_area)
        {
            max_area = component_area;
            max_label_component = i;
        }
    }

    // Suppress (mask set to 0) all components with non greatest area
    for (int row = 0; row < src.rows; row++)
    {
        for (int col = 0; col < src.cols; col++)
        {
            if (connected_components_labels.at<int>(row, col) != max_label_component)
            {
                dst.at<uchar>(row, col) = 0;
            }
        }
    }
}

double playing_field_localizer::angular_coefficient(const Point &p1, const Point &p2)
{
    return (p2.y - p1.y) / (p2.x - p1.x);
}

bool playing_field_localizer::is_vertical_line(const Point &p1, const Point &p2)
{
    return p1.x == p2.x;
}

bool playing_field_localizer::are_parallel_lines(double m1, double m2)
{
    const float EPSILON = 0.001;
    return abs(m1 - m2) <= EPSILON;
}

double playing_field_localizer::intercept(const Point &p1, const Point &p2)
{
    return p1.y - p1.x * angular_coefficient(p1, p2);
}

bool playing_field_localizer::is_within_image(const Point &p, int rows, int cols)
{
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

// Finds the intersection of two lines.
// The lines are defined by (o1, p1) and (o2, p2).
bool playing_field_localizer::intersection(Point o1, Point p1, Point o2, Point p2, Point &r, int rows, int cols)
{
    Point x = o2 - o1;
    Point d1 = p1 - o1;
    Point d2 = p2 - o2;

    const float EPSILON = 1e-8;
    float cross = d1.x * d2.y - d1.y * d2.x;
    if (abs(cross) < EPSILON)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x) / cross;
    r = o1 + d1 * t1;

    return is_within_image(r, rows, cols);
}

// Finds the intersection of two lines.
// The lines are defined by (o1, p1) and (o2, p2).
void playing_field_localizer::intersections(const vector<vector<Point>> &points, vector<Point> &inters, int rows, int cols)
{
    for (int i = 0; i < points.size() - 1; i++)
    {
        for (int j = i + 1; j <= points.size() - 1; j++)
        {
            Point inte;
            if (intersection(points[i][0], points[i][1], points[j][0], points[j][1], inte, rows, cols))
                inters.push_back(inte);
        }
    }
}

double playing_field_localizer::angle_between_lines(double m1, double m2)
{
    double angle = atan(abs((m1 - m2) / (1 + m1 * m2)));
    if (angle >= 0)
        return angle;
    else
        return angle + CV_PI;
}

void playing_field_localizer::draw_pool_table(vector<Point> inters, Mat &image)
{
    if (is_vertical_line(inters[0], inters[1]) ||
        is_vertical_line(inters[0], inters[2]) ||
        is_vertical_line(inters[0], inters[3]))
    {
        vector<int> x_coord = {inters[0].x, inters[1].x, inters[2].x, inters[3].x};
        vector<int> y_coord = {inters[0].y, inters[1].y, inters[2].y, inters[3].y};

        int x1 = *min_element(x_coord.begin(), x_coord.end()); // top-left pt. is the leftmost of the 4 points
        int x2 = *max_element(x_coord.begin(), x_coord.end()); // bottom-right pt. is the rightmost of the 4 points
        int y1 = *min_element(y_coord.begin(), y_coord.end()); // top-left pt. is the uppermost of the 4 points
        int y2 = *max_element(y_coord.begin(), y_coord.end()); // bottom-right pt. is the lowermost of the 4 points

        rectangle(image, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 1);
    }

    else
    {
        double m1 = angular_coefficient(inters[0], inters[1]); // line 1
        double m2 = angular_coefficient(inters[0], inters[2]); // line 2
        double m3 = angular_coefficient(inters[0], inters[3]); // line 3

        double theta1 = angle_between_lines(m1, m2); // angle between line 1 and line 2
        double theta2 = angle_between_lines(m1, m3); // angle between line 1 and line 3
        double theta3 = angle_between_lines(m2, m3); // angle between line 2 and line 3

        if (theta1 >= theta2 && theta1 >= theta3)
        {
            line(image, inters[0], inters[1], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[0], inters[2], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[3], inters[1], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[3], inters[2], Scalar(0, 0, 255), 1, LINE_AA);
        }
        else if (theta2 >= theta1 && theta2 >= theta3)
        {
            line(image, inters[0], inters[1], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[0], inters[3], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[2], inters[1], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[2], inters[3], Scalar(0, 0, 255), 1, LINE_AA);
        }
        else
        {
            line(image, inters[0], inters[2], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[0], inters[3], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[1], inters[2], Scalar(0, 0, 255), 1, LINE_AA);
            line(image, inters[1], inters[3], Scalar(0, 0, 255), 1, LINE_AA);
        }
    }
}

void playing_field_localizer::get_pairs_points_per_line(const vector<Vec3f> &lines, vector<vector<Point>> &points)
{
    // Arbitrary x coordinate to compute the 2 points in each line.
    const float POINT_X = 1000;

    for (auto line : lines)
    {
        float rho = line[0];
        float theta = line[1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + POINT_X * (-b));
        pt1.y = cvRound(y0 + POINT_X * (a));
        pt2.x = cvRound(x0 - POINT_X * (-b));
        pt2.y = cvRound(y0 - POINT_X * (a));

        points.push_back({pt1, pt2});
    }
}

void playing_field_localizer::sort_points_clockwise(vector<Point> &points)
{
    Point center;
    for (auto point : points)
        center += point;

    center.x /= points.size();
    center.y /= points.size();

    sort(points.begin(), points.end(), [center](const Point &pt1, const Point &pt2)
         {
        if (pt1.x - center.x >= 0 && pt2.x - center.x < 0)
            return false;
        if (pt1.x - center.x < 0 && pt2.x - center.x >= 0)
            return true;
        if (pt1.x - center.x == 0 && pt2.x - center.x == 0) 
        {
            if (pt1.y - center.y >= 0 || pt2.y - center.y >= 0)
                return pt1.y < pt2.y;
            return pt2.y > pt1.y;
        }

        // Compute cross product between pt1-center and pt2-center. If it is < 0, then pt1 comes first in
        // clockwise order. If it is > 0, then pt2 comes first.
        int cross_product = (pt1.x - center.x) * (pt2.y - center.y) - (pt2.x - center.x) * (pt1.y - center.y);
        return cross_product > 0; });
}
