#include "playing_field_localizer.h"

using namespace cv;
using namespace std;

void playing_field_localizer::hough_approach(cv::Mat src, cv::Mat &dst)
{
    Mat cdst, cdstP;

    bilateralFilter(src.clone(), src, 9, 75, 75);
    Canny(src, dst, 50, 200, 3);

    imshow("", dst);
    waitKey(0);

    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    // Standard Hough Line Transform
    vector<Vec2f> lines;
    HoughLines(dst, lines, 1.7, CV_PI / 270, 250, 0, 0);
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
        line(src, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    // Show results
    imshow("", src);
    // imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    // imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    waitKey();
}

void playing_field_localizer::localize(cv::Mat src, cv::Mat &dst)
{
    hough_approach(src, dst);
}