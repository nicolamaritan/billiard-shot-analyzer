// Author: Francesco Boscolo Meneguolo 2119969

#include "playing_field_localization.h"
#include "geometry.h"
#include "segmentation.h"

#include <iostream>
#include <cmath>
#include <map>
#include <limits>

using namespace cv;
using namespace std;

void playing_field_localizer::localize(const Mat &src)
{
    const int FILTER_SIZE = 3;
    const int FILTER_SIGMA = 20;
    Mat blurred;
    GaussianBlur(src.clone(), blurred, Size(FILTER_SIZE, FILTER_SIZE), FILTER_SIGMA, FILTER_SIGMA);

    Mat segmented, labels;
    segmentation(blurred, segmented);

    const int RADIUS = 30;
    Vec3b board_color = get_playing_field_color(segmented, RADIUS);

    Mat mask;
    inRange(segmented, board_color, board_color, mask);
    segmented.setTo(Scalar(0, 0, 0), mask);

    non_maxima_connected_component_suppression(mask.clone(), mask);

    const int THRESHOLD_1_CANNY = 50;
    const int THRESHOLD_2_CANNY = 150;
    Mat edges;
    Canny(mask, edges, THRESHOLD_1_CANNY, THRESHOLD_2_CANNY);

    vector<Vec3f> lines, refined_lines;
    find_lines(edges, lines);
    refine_lines(lines, refined_lines);

    draw_lines(edges, refined_lines);

    vector<Point> refined_lines_intersections;
    intersections(refined_lines, refined_lines_intersections, src.rows, src.cols);

    sort_points_clockwise(refined_lines_intersections);
    localization.corners = refined_lines_intersections;

    vector<Point> hole_points;
    estimate_holes_location(hole_points);
    localization.hole_points = hole_points;

    Mat table_mask(Size(src.cols, src.rows), CV_8U);
    table_mask.setTo(0);
    fillConvexPoly(table_mask, refined_lines_intersections, 255);
    localization.mask = table_mask;
}

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

    kmeans(dst.clone(), dst, 3);
}

void playing_field_localizer::find_lines(const Mat &edges, vector<Vec3f> &lines)
{
    const float RHO_RESOLUTION = 1.5;   // In pixels.
    const float THETA_RESOLUTION = 1.8; // In radians.
    const int THRESHOLD = 110;

    Mat cdst;
    cvtColor(edges, cdst, COLOR_GRAY2BGR);
    HoughLines(edges, lines, RHO_RESOLUTION, THETA_RESOLUTION * CV_PI / 180, THRESHOLD, 0, 0);
}

void playing_field_localizer::refine_lines(const vector<Vec3f> &lines, vector<Vec3f> &refined_lines)
{
    const float RHO_THRESHOLD = 40;
    const float THETA_THRESHOLD = 0.5;
    vector<Vec3f> lines_copy = lines;

    while (!lines_copy.empty())
    {
        Vec3f reference_line = lines_copy.back();
        lines_copy.pop_back();
        vector<Vec3f> similar_lines;

        dump_similar_lines(reference_line, lines_copy, similar_lines, RHO_THRESHOLD, THETA_THRESHOLD);

        // Compute a new mean line with the dumped ones
        Vec3f mean_line;
        int total_votes = 0;
        for (Vec3f similar_line : similar_lines)
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
    const float ARBITRARY_COORDINATE = 1000; // Arbitrary constant for line plotting

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + ARBITRARY_COORDINATE * (-b));
        pt1.y = cvRound(y0 + ARBITRARY_COORDINATE * (a));
        pt2.x = cvRound(x0 - ARBITRARY_COORDINATE * (-b));
        pt2.y = cvRound(y0 - ARBITRARY_COORDINATE * (a));
        const Scalar RED = Scalar(0, 255, 0);
        line(src_bgr, pt1, pt2, RED, 1, LINE_AA);
    }
}

void playing_field_localizer::dump_similar_lines(const Vec3f &reference_line, vector<Vec3f> &lines, vector<Vec3f> &similar_lines, float rho_threshold, float theta_threshold)
{
    similar_lines.push_back(reference_line);

    // Insert into similar_lines all the similar lines and removes them from lines.
    int i = 0;
    while (i < lines.size())
    {
        Vec3f line = lines.at(i);
        if (abs(line[0] - reference_line[0]) < rho_threshold && abs(line[1] - reference_line[1]) < theta_threshold)
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

bool playing_field_localizer::is_within_image(const Point &p, int rows, int cols)
{
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

void playing_field_localizer::sort_points_clockwise(vector<Point> &points)
{
    Point center;
    for (Point point : points)
        center += point;

    center.x /= points.size();
    center.y /= points.size();

    sort(points.begin(), points.end(), [center](const Point &pt1, const Point &pt2)
         {
        /*
            This conditions allow to totally order set of points in clockwise order.
            In fact, with the expression below, the sort function could loop indefinitely while
            trying to compare pairs of points.
        */
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

        /*
            Compute cross product between pt1-center and pt2-center. If it is < 0, then pt1 comes first in
            clockwise order. If it is > 0, then pt2 comes first.
        */
        int cross_product = (pt1.x - center.x) * (pt2.y - center.y) - (pt2.x - center.x) * (pt1.y - center.y);
        return cross_product > 0; });
}

void playing_field_localizer::estimate_holes_location(vector<Point> &hole_points)
{
    vector<Point> corners = localization.corners;
    pair<Point, Point> positive_diagonal = {corners.at(0), corners.at(2)};
    pair<Point, Point> negative_diagonal = {corners.at(1), corners.at(3)};
    Point playing_field_center;
    bool is_perspective_view = false;
    intersection(positive_diagonal, negative_diagonal, playing_field_center);

    // Computation of long and short edges.
    pair<Point, Point> short_edge, long_edge_1, long_edge_2;

    // If the two angular coefficients have similar absolute value and opposite sign, then we have a perspective view
    const float ANGULAR_COEFFICIENT_EPS = 0.01;
    if (abs(angular_coefficient(positive_diagonal) + angular_coefficient(negative_diagonal)) < ANGULAR_COEFFICIENT_EPS)
    {
        is_perspective_view = true;
        long_edge_1 = {corners.at(0), corners.at(1)};
        long_edge_2 = {corners.at(3), corners.at(2)};
        short_edge = {corners.at(3), corners.at(0)};
    }
    else
    {
        if (norm(corners.at(0) - corners.at(1)) < norm(corners.at(1) - corners.at(2)))
        {
            short_edge = {corners.at(0), corners.at(1)};
            long_edge_1 = {corners.at(1), corners.at(2)};
            long_edge_2 = {corners.at(3), corners.at(0)};
        }
        else
        {
            short_edge = {corners.at(1), corners.at(2)};
            long_edge_1 = {corners.at(0), corners.at(1)};
            long_edge_2 = {corners.at(2), corners.at(3)};
        }
    }

    /*
        A line of the same direction of the short edge intersects the two long
        edges in the hole positions. We now find such intersections.
    */
    Point short_edge_offset = short_edge.first - short_edge.second;
    Point lateral_hole_1, lateral_hole_2;
    intersection({playing_field_center, playing_field_center + short_edge_offset}, long_edge_1, lateral_hole_1);
    intersection({playing_field_center, playing_field_center + short_edge_offset}, long_edge_2, lateral_hole_2);

    /*
        We now employ float representation of points for precise computation of the refined
        holes points. In fact, using cv::Point we obtain non negligible truncating errors.
    */
    Point2f bottom_left = corners.at(0);
    Point2f top_left = corners.at(1);
    Point2f top_right = corners.at(2);
    Point2f bottom_right = corners.at(3);

    Point2f playing_field_center_float = static_cast<Point2f>(playing_field_center);
    Point2f lateral_hole_1_float = static_cast<Point2f>(lateral_hole_1);
    Point2f lateral_hole_2_float = static_cast<Point2f>(lateral_hole_2);
    Point2f top_left_float = static_cast<Point2f>(top_left);
    Point2f top_right_float = static_cast<Point2f>(top_right);
    Point2f bottom_left_float = static_cast<Point2f>(bottom_left);
    Point2f bottom_right_float = static_cast<Point2f>(bottom_right);

    /*
        Refined hole points computation
        Refined hole points move "a bit towards" the playing field center.
        The amount is defined by the "adjustment" vars below.
        This is done to better estimate their location.
    */
    const float LATERAL_HOLES_ADJUSTMENT = 15;
    const float BOTTOM_CORNERS_ADJUSTMENT = is_perspective_view ? 25 : 10;
    const float TOP_CORNERS_ADJUSTMENT = is_perspective_view ? 15 : 10;

    Point2f lateral_hole_1_refined = lateral_hole_1_float + ((playing_field_center_float - lateral_hole_1_float) / norm(playing_field_center_float - lateral_hole_1_float)) * LATERAL_HOLES_ADJUSTMENT;
    Point2f lateral_hole_2_refined = lateral_hole_2_float + ((playing_field_center_float - lateral_hole_2_float) / norm(playing_field_center_float - lateral_hole_2_float)) * LATERAL_HOLES_ADJUSTMENT;

    Point2f top_left_refined = top_left_float + ((playing_field_center_float - top_left_float) / norm(playing_field_center_float - top_left_float)) * TOP_CORNERS_ADJUSTMENT;
    Point2f top_right_refined = top_right_float + ((playing_field_center_float - top_right_float) / norm(playing_field_center_float - top_right_float)) * TOP_CORNERS_ADJUSTMENT;

    Point2f bottom_left_refined = bottom_left_float + ((playing_field_center_float - bottom_left_float) / norm(playing_field_center_float - bottom_left_float)) * BOTTOM_CORNERS_ADJUSTMENT;
    Point2f bottom_right_refined = bottom_right_float + ((playing_field_center_float - bottom_right_float) / norm(playing_field_center_float - bottom_right_float)) * BOTTOM_CORNERS_ADJUSTMENT;

    hole_points.push_back(static_cast<Point>(lateral_hole_1_refined));
    hole_points.push_back(static_cast<Point>(lateral_hole_2_refined));
    hole_points.push_back(static_cast<Point>(top_left_refined));
    hole_points.push_back(static_cast<Point>(top_right_refined));
    hole_points.push_back(static_cast<Point>(bottom_left_refined));
    hole_points.push_back(static_cast<Point>(bottom_right_refined));
}