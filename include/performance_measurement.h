//Author: Eddie Carraro

#ifndef PERFORMANCE_MEASUREMENT
#define PERFORMANCE_MEASUREMENT

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class performance_measurement
{
    public:
    float balls_detection_performance(int x, int y, int width, int height, int ball_ID);
    float balls_segmentation_performance();
    float table_segmentation_performance(cv::Mat ground_truth_mask, cv::Mat found_mask);

    private:

};


#endif