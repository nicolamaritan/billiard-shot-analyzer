#include "utils.h"

using namespace std;
using namespace cv;

// Finds the intersection of two lines.
// The lines are defined by (o1, pt1) and (o2, pt2).
void intersections(const vector<Vec3f> &lines, vector<Point> &out_intersections, int rows, int cols)
{
    // It is easier to compute intersections by expressing one line as a pair (pt1, pt2),
    // distinct points in the line.
    vector<pair<Point, Point>> points;
    get_pairs_points_per_line(lines, points);

    for (int i = 0; i < points.size() - 1; i++)
    {
        for (int j = i + 1; j < points.size(); j++)
        {
            Point intersection_ij;
            if (intersection(points.at(i), points.at(j), intersection_ij, rows, cols))
                out_intersections.push_back(intersection_ij);
        }
    }
}

void get_pairs_points_per_line(const vector<Vec3f> &lines, vector<pair<Point, Point>> &pts)
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

        pts.push_back({pt1, pt2});
    }
}

// Finds the intersection of two lines.
// The lines are defined by (o1, pt1) and (o2, pt2).
bool intersection(pair<Point, Point> pts_line_1, pair<Point, Point> pts_line_2, Point &intersection_pt, int rows, int cols)
{
    // pts_line_1 -> {(x1,y1), (x2,y2)}
    // pts_line_2 -> {(x3,y3), (x4,y4)}
    // We follow the formula in https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    Point difference_12 = pts_line_1.first - pts_line_1.second;
    Point difference_13 = pts_line_1.first - pts_line_2.first;
    Point difference_34 = pts_line_2.first - pts_line_2.second;

    const float EPSILON = 1e-8;
    float cross_product = difference_12.x * difference_34.y - difference_12.y * difference_34.x;
    if (abs(cross_product) < EPSILON)
        return false;

    // t is a real number in the formula computed as follows
    double t = (difference_13.x * difference_34.y - difference_13.y * difference_34.x) / cross_product;
    
    intersection_pt = pts_line_1.first - difference_12 * t;

    return is_within_image(intersection_pt, rows, cols);
}

bool is_within_image(const Point &p, int rows, int cols)
{
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}