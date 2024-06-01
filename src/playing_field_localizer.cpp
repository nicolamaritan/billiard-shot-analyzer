#include <iostream>
#include <cmath>
#include <map>
#include "playing_field_localizer.h"

using namespace cv;
using namespace std;


void playing_field_localizer::segmentation(const Mat &src, Mat &dst)
{
    Mat src_hsv;
    cvtColor(src, src_hsv, COLOR_BGR2HSV);
    dst = src_hsv;

    Mat data;
    dst.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // do kmeans
    Mat labels, centers;
    kmeans(data, 4, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 1.0), 3,
           KMEANS_PP_CENTERS, centers);

    // reshape both to a single row of Vec3f pixels:
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

    // replace pixel values with their center value:
    Vec3f *p = data.ptr<Vec3f>();
    for (size_t i = 0; i < data.rows; i++)
    {
        int center_id = labels.at<int>(i);

        if (center_id == 0)
        {
            p[i] = centers.at<Vec3f>(center_id);
        }
        else
        {
            p[i] = Vec3f(0, 0, 0);
        }
        p[i] = centers.at<Vec3f>(center_id);
    }

    dst = data.reshape(3, dst.rows);
    dst.convertTo(dst, CV_8U);
}

cv::Vec3b playing_field_localizer::get_board_color(const cv::Mat &src)
{
    return src.at<Vec3b>(src.rows / 2, src.cols / 2);
}

double playing_field_localizer::angular_coeff(const Point &p1, const Point &p2)
{
    return (p2.y - p1.y)/(p2.x - p1.x);
}

bool playing_field_localizer::is_vertical_line(const Point &p1, const Point &p2)
{
    return p1.x == p2.x;
}

bool playing_field_localizer::are_parallel_lines(double m1, double m2)
{
    return abs(m1-m2) <= 0.001;
}

double playing_field_localizer::intercept(const Point &p1, const Point &p2)
{
    return p1.y - p1.x * angular_coeff(p1, p2);
}

bool playing_field_localizer::is_within_image(const Point &p, int rows, int cols)
{
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

// Finds the intersection of two lines.
// The lines are defined by (o1, p1) and (o2, p2).
bool playing_field_localizer::intersection(cv::Point o1, cv::Point p1, cv::Point o2, cv::Point p2, cv::Point &r)
{
    cv::Point x = o2 - o1;
    cv::Point d1 = p1 - o1;
    cv::Point d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;
    
    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

// Finds the intersection of two lines.
// The lines are defined by (o1, p1) and (o2, p2).
void playing_field_localizer::intersections(const vector<vector<Point>> &points, vector<Point> &inters, int rows, int cols, Mat& img)
{
    for(int i = 0; i < points.size()-1; i++)
    {
        for(int j = i+1; j <= points.size()-1; j++)
        {
            Point inte;
            if(intersection(points[i][0], points[i][1], points[j][0], points[j][1], inte))
                inters.push_back(inte);
        }
    }
    cout << "Numero di punti di intersezione: " << inters.size() << endl;
}

double playing_field_localizer::angle_between_lines(double m1, double m2)
{
    return atan(abs((m1-m2)/(1+m1*m2)));
}

void playing_field_localizer::draw_pool_table(vector<Point> inters, Mat& image)
{
    cout << "Numero di punti di intersezione: " << inters.size() << endl;

    cout << "Stampo i punti di intersezione:" << endl;
    cout << "x = " << inters[0].x << ", " << " y = " << inters[0].y << endl;
    cout << "x = " << inters[1].x << ", " << " y = " << inters[1].y << endl;
    cout << "x = " << inters[2].x << ", " << " y = " << inters[2].y << endl;
    cout << "x = " << inters[3].x << ", " << " y = " << inters[3].y << endl;

    if(is_vertical_line(inters[0], inters[1]) ||
        is_vertical_line(inters[0], inters[2]) ||
        is_vertical_line(inters[0], inters[3]))
    {
        vector<int> x_coord = {inters[0].x, inters[1].x, inters[2].x, inters[3].x};
        vector<int> y_coord = {inters[0].y, inters[1].y, inters[2].y, inters[3].y};
        
        int x1 = *min_element(x_coord.begin(), x_coord.end());//top-left pt. is the leftmost of the 4 points
        int x2 = *max_element(x_coord.begin(), x_coord.end());//bottom-right pt. is the rightmost of the 4 points
        int y1 = *min_element(y_coord.begin(), y_coord.end());//top-left pt. is the uppermost of the 4 points
        int y2 = *max_element(y_coord.begin(), y_coord.end());//bottom-right pt. is the lowermost of the 4 points

        rectangle(image, Point(x1,y1), Point(x2,y2), Scalar(0, 0, 255), 3);
    }
    
    else
    {
        double m1 = angular_coeff(inters[0], inters[1]); //line 1
        double m2 =  angular_coeff(inters[0], inters[2]); // line 2
        double m3 = angular_coeff(inters[0], inters[3]); // line 3

        double theta1 = angle_between_lines(m1, m2); // angle between line 1 and line 2
        double theta2 = angle_between_lines(m1, m3); // angle between line 1 and line 3
        double theta3 = angle_between_lines(m2, m3); // angle between line 2 and line 3

        if(theta1 >= theta2 && theta1 >= theta3)
        {
            line(image, inters[0], inters[1], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[0], inters[2], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[3], inters[1], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[3], inters[2], Scalar(0, 0, 255), 3, LINE_AA);
        }
        else if(theta2 >= theta1 && theta2 >= theta3)
        {
            line(image, inters[0], inters[1], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[0], inters[3], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[2], inters[1], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[2], inters[3], Scalar(0, 0, 255), 3, LINE_AA);
        }
        else
        {
            line(image, inters[0], inters[2], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[0], inters[3], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[1], inters[2], Scalar(0, 0, 255), 3, LINE_AA);
            line(image, inters[1], inters[3], Scalar(0, 0, 255), 3, LINE_AA);
        }
    }

}
void playing_field_localizer::find_lines(const cv::Mat &edges)
{
    Mat cdst;

    // Copy edges to the images that will display the results in BGR
    cvtColor(edges, cdst, COLOR_GRAY2BGR);
    // Standard Hough Line Transform
    vector<Vec2f> lines;                                 // will hold the results of the detection
    HoughLines(edges, lines, 1.6, 1.8 * CV_PI / 180, 120, 0, 0); // runs the actual detection
    // Draw the lines
    vector<vector<Point>> points;
    vector<Point> inters;
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
        points.push_back({pt1, pt2});
        //line(cdst, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
    }
    //imshow("1", cdst);
    
    intersections(points, inters, cdst.rows, cdst.cols, cdst);
    draw_pool_table(inters, cdst);
    imshow("POOL TABLE", cdst);
    
    

    waitKey(0);
}

void playing_field_localizer::localize(const Mat &src, Mat &dst)
{
    GaussianBlur(src.clone(), src, Size(3, 3), 12, 12);

    Mat segmented, labels;
    segmentation(src, segmented);

    imshow("", segmented);
    waitKey(0);

    Vec3b board_color = get_board_color(segmented);

    Mat mask;
    inRange(segmented, board_color, board_color, mask);
    segmented.setTo(Scalar(0, 0, 0), mask);

    imshow("", mask);
    waitKey(0);

    Mat element = getStructuringElement(MORPH_CROSS, Size(5, 5));
    morphologyEx(mask.clone(), mask, MORPH_OPEN, element);
    imshow("", mask);
    waitKey(0);

    element = getStructuringElement(MORPH_RECT, Size(20, 20));
    morphologyEx(mask.clone(), mask, MORPH_CLOSE, element);
    imshow("", mask);
    waitKey(0);

    Mat edges;
    Canny(mask, edges, 50, 150);
    imshow("", edges);
    waitKey(0);

    find_lines(edges);
}