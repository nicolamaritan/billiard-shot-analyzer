#include "frame_segmentation.h"
#include "balls_localization.h"
#include "playing_field_localization.h"
#include "performance_measurement.h"

using namespace cv;
using namespace std;

/**
 * @brief Segments the frame based on color information.
 *
 * This function applies color-based segmentation to the input frame, preserving the background if specified.
 *
 * @param src The source image to be segmented.
 * @param dst The destination image where the segmentation result will be stored.
 * @param frame_segmentation A matrix containing segmentation labels for the frame.
 * @param preserve_background Boolean flag to indicate if the background should be preserved in the segmentation.
 */
void color_segmentation(const cv::Mat &src, cv::Mat &dst, const cv::Mat &frame_segmentation, bool preserve_background);



void get_colored_frame_segmentation(const Mat &src, Mat &dst, bool preserve_background)
{
    playing_field_localizer plf_loc;
    plf_loc.localize(src);
    balls_localizer blls_loc(plf_loc.get_localization());
    blls_loc.localize(src);

    Mat frame_segmentation;
    get_frame_segmentation(src, frame_segmentation);

    color_segmentation(src, dst, frame_segmentation, preserve_background);
    
    // Draw yellow lines
    vector<Point> corners = plf_loc.get_localization().corners;
    for (size_t i = 0; i < corners.size(); i++)
    {
        const Scalar YELLOW_COLOR = Scalar(0, 255, 255);
        const int LINE_THICKNESS = 3;
        line(dst, corners[i], corners[(i + 1) % corners.size()], YELLOW_COLOR, LINE_THICKNESS);
    }
}



void color_segmentation(const cv::Mat &src, cv::Mat &dst, const cv::Mat &frame_segmentation, bool preserve_background)
{
    dst = src.clone();

    // BGR color mapping
    vector<cv::Vec3b> color_map(6);
    color_map.at(0) = cv::Vec3b(128, 128, 128); // Gray
    color_map.at(1) = cv::Vec3b(255, 255, 255); // White
    color_map.at(2) = cv::Vec3b(0, 0, 0);       // Black
    color_map.at(3) = cv::Vec3b(0, 0, 255);     // Red
    color_map.at(4) = cv::Vec3b(255, 0, 0);     // Blue
    color_map.at(5) = cv::Vec3b(0, 255, 0);     // Green

    // background id is excluded when preserve_color is true
    const int MIN_COLOR = preserve_background ? 1 : 0;
    const int MAX_COLOR = 6;

    for (int i = 0; i < frame_segmentation.rows; i++)
    {
        for (int j = 0; j < frame_segmentation.cols; j++)
        {
            // Color based on label id
            uchar pixel_value = frame_segmentation.at<uchar>(i, j);
            if (pixel_value >= MIN_COLOR && pixel_value < MAX_COLOR)
                dst.at<cv::Vec3b>(i, j) = color_map.at(pixel_value);
        }
    }
}



void get_frame_segmentation(const Mat &src, Mat &dst)
{
    playing_field_localizer plf_localizer;
    plf_localizer.localize(src);
    playing_field_localization plf_localization = plf_localizer.get_localization();

    balls_localizer blls_localizer(plf_localization);
    blls_localizer.localize(src);
    balls_localization blls_localization = blls_localizer.get_localization();

    Mat segmentation(src.size(), CV_8UC1);
    segmentation.setTo(Scalar(label_id::background));
    segmentation.setTo(Scalar(label_id::playing_field), plf_localization.mask);

    Vec3f cue_circle = blls_localization.cue.circle;
    circle(segmentation, Point(cue_circle[0], cue_circle[1]), cue_circle[2], Scalar(label_id::cue), FILLED);

    Vec3f black_circle = blls_localization.black.circle;
    circle(segmentation, Point(black_circle[0], black_circle[1]), black_circle[2], Scalar(label_id::black), FILLED);

    for (ball_localization loc : blls_localization.solids)
    {
        Vec3f loc_circle = loc.circle;
        circle(segmentation, Point(loc_circle[0], loc_circle[1]), loc_circle[2], Scalar(label_id::solids), FILLED);
    }

    for (ball_localization loc : blls_localization.stripes)
    {
        Vec3f loc_circle = loc.circle;
        circle(segmentation, Point(loc_circle[0], loc_circle[1]), loc_circle[2], Scalar(label_id::stripes), FILLED);
    }

    dst = segmentation;
}