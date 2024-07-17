#include "frame_detection.h"
#include "playing_field_localization.h"
#include "balls_localization.h"

using namespace cv;
using namespace std;

void draw_transparent_rect(Mat &src, Rect rect, Scalar color, double alpha);

void get_frame_detection(const Mat &src, Mat &dst)
{
    dst = src.clone();

    playing_field_localizer plf_localizer;
    plf_localizer.localize(src);

    balls_localizer blls_localizer(plf_localizer.get_localization());
    blls_localizer.localize(src);
    balls_localization blls_localization = blls_localizer.get_localization();

    const float ALPHA = 0.4;
    const Scalar WHITE = Scalar(255, 255, 255);
    const Scalar BLACK = Scalar(0, 0, 0);
    const Scalar BLUE = Scalar(255, 0, 0);
    const Scalar RED = Scalar(0, 0, 255);

    draw_transparent_rect(dst, blls_localization.cue.bounding_box, WHITE, ALPHA);
    draw_transparent_rect(dst, blls_localization.black.bounding_box, BLACK, ALPHA);

    for (ball_localization localization : blls_localization.solids)
    {
        draw_transparent_rect(dst, localization.bounding_box, BLUE, ALPHA);
    }

    for (ball_localization localization : blls_localization.stripes)
    {
        draw_transparent_rect(dst, localization.bounding_box, RED, ALPHA);
    }

    // Draw yellow lines
    vector<Point> corners = plf_localizer.get_localization().corners;
    for (size_t i = 0; i < corners.size(); i++)
    {
        const Scalar YELLOW_COLOR = Scalar(0, 255, 255);
        const int LINE_THICKNESS = 3;
        line(dst, corners[i], corners[(i + 1) % corners.size()], YELLOW_COLOR, LINE_THICKNESS);
    }
}

void draw_transparent_rect(Mat &image, Rect rect, Scalar color, double alpha) 
{
    Mat overlay;
    image.copyTo(overlay);
    rectangle(overlay, rect, color, FILLED);
    addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image);

    const int THICKNESS = 2;
    rectangle(image, rect, color, THICKNESS);
}