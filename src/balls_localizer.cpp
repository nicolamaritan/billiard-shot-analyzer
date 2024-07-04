#include <iostream>
#include <cmath>
#include <map>
#include "balls_localizer.h"
#include "utils.h"


using namespace cv;
using namespace std;


void balls_localizer::localize_red_balls(const Mat &src, Mat &dst)
{
    Mat src_hsv;
    cvtColor(src, src_hsv, COLOR_BGR2HSV);

    // Apply uniform Value (of HSV) for the whole image, to handle different brightnesses    
    const int VALUE_UNIFORM = 128; 
    vector<Mat> hsv_channels;
    split(src_hsv, hsv_channels);
    hsv_channels[2].setTo(VALUE_UNIFORM);
    merge(hsv_channels, src_hsv);
    cvtColor(src_hsv, src, COLOR_HSV2BGR);
    Vec3b red_reference_lower = Vec3b(0, 0, 150);
    Vec3b red_reference_upper = Vec3b(255, 255, 255);
    inRange(src, red_reference_lower, red_reference_upper, dst);
    cvtColor(dst.clone(), dst, COLOR_GRAY2BGR);
    Mat a(Size(src.rows, src.cols * 2), CV_8UC3);
    hconcat(src, dst.clone(), a);
    imshow("", a);
    waitKey(0);
}

void balls_localizer::localize_balls(const Mat &src, Mat &dst, const Mat temp_edges, vector<Point> corners)
{
    Mat mask;
    //cvtColor(src, dst, COLOR_BGR2YCrCb);
    //imshow("", dst);
    //waitKey(0);
    create_table_mask(src, mask, corners);
    //apply_mask(dst, mask);
    Mat red_mask;
    localize_red_balls(src, red_mask);
    /*
    vector<Vec3f> circles;
    Mat dst_gray;
    cvtColor(dst, dst_gray, COLOR_BGR2GRAY);
    HoughCircles(dst_gray, circles, cv::HOUGH_GRADIENT, 5, 40, 160, 0.8, 5, 15);
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // draw the circle center
        circle( dst, center, 3, Scalar(0,255,0));
        // draw the circle outline
        circle( dst, center, radius, Scalar(0,0,255));
    }
    */
    //imshow( "", dst);
    //waitKey(0);


    /*
    Mat mask, gray;
    dst = src.clone();
    create_table_mask(src, mask, corners);
    Mat masked_image = src.clone();
    apply_mask(masked_image, mask);
    
    imshow( "", masked_image);
    waitKey(0);
    cvtColor(masked_image, gray, COLOR_BGR2GRAY);
    imshow("", gray);
    waitKey(0);
    Mat detected_edges;

    Canny( gray, detected_edges, 50, 150, 3);
    imshow("", detected_edges);
    waitKey(0);
    
    dst = temp_edges;
    cvtColor(dst.clone(), dst, COLOR_GRAY2BGR);
    vector<Vec3f> circles;
    HoughCircles(temp_edges, circles, cv::HOUGH_GRADIENT, 5, 40, 160, 0.8, 5, 15);
    cout << "Here" << endl;
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // draw the circle center
        circle( dst, center, 3, Scalar(0,255,0));
        // draw the circle outline
        circle( dst, center, radius, Scalar(0,0,255));
    }
    
    imshow( "", dst);
    waitKey(0);
    */


}

void balls_localizer::create_table_mask(const Mat &src, Mat &mask, vector<Point> corners)
{
    mask = Mat::zeros(src.rows, src.cols, CV_8U);
    fillConvexPoly(mask, corners, Scalar(255));

}

bool balls_localizer::is_same_color(const Vec3b& color1, const Vec3b& color2)
{
    return (abs(color1[0] - color2[0]) <= 60) && (abs(color1[1] - color2[1]) <= 60) 
            && (abs(color1[2] == color2[2]) <= 60);
}

void balls_localizer::process_table(Mat &image, const Vec3b& color, int row, int col)
{

    if(is_same_color(image.at<Vec3b>(row, col), color))
        image.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
    
    
}


void balls_localizer::apply_mask(Mat &image, const Mat &mask)
{
    const int RADIUS = 30;
    Vec3b board_color = get_board_color(image, RADIUS);
    for(int i = 0; i < image.rows; i++)
    {
        for(int j = 0; j < image.cols; j++)
        {
            
            if(mask.at<uchar>(i,j) == 255)
                process_table(image, board_color, i, j);
            else
            {
            //if(mask.at<uchar>(i,j) == 0)
                image.at<Vec3b>(i,j) = Vec3b(0, 0, 0);
            }
        }
    }


}