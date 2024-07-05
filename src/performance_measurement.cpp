//Author: Eddie Carraro

#include "performance_measurement.h"
#include <iostream>

using namespace cv;
using namespace std;

//typedef Point3_<unsigned char> Pixel;


float performance_measurement::balls_detection_performance(int x, int y, int width, int height, int ball_ID)
{
    return 0.0f;
}

float performance_measurement::balls_segmentation_performance()
{
    return 0.0f;
}

float performance_measurement::table_segmentation_performance(Mat ground_truth_mask, Mat found_mask)
{
    float intersection_of_masks = 0;    // Number of intersections
    float union_of_masks = 0;           // Number of total pixels of the union

    //maybe we can use forEach???
    for(int i = 0; i < ground_truth_mask.rows; i++)
    {
        for(int j = 0; j < ground_truth_mask.cols; j++)
        {
            //intersection
            if((ground_truth_mask.at<Vec3b>(i,j) != Vec3b(0,0,0)) && (found_mask.at<Vec3b>(i,j) != Vec3b(0,0,0)))
            {
                intersection_of_masks++;
            }

            if((ground_truth_mask.at<Vec3b>(i,j) != Vec3b(0,0,0)) || (found_mask.at<Vec3b>(i,j) != Vec3b(0,0,0)))
            {
                union_of_masks++;
            }
        }
    }

    return intersection_of_masks / union_of_masks;
}