#include "frame_segmentation.h"
#include "playing_field_localization.h"
#include "balls_localization.h"

using namespace cv;

void frame_segmentation(const Mat& src, Mat& dst)
{
    playing_field_localizer plf_localizer;
    plf_localizer.localize(src);
    playing_field_localization plf_localization = plf_localizer.get_localization();

    balls_localizer blls_localizer(plf_localization);
    blls_localizer.localize(src);
    balls_localization blls_localization = blls_localizer.get_localization();

    Mat segmentation(src.size(), CV_8UC1);
    segmentation.setTo(Scalar(segmentation_label::background));
    segmentation.setTo(Scalar(segmentation_label::playing_field), plf_localization.mask);

    Vec3f cue_circle = blls_localization.cue.circle;
    circle(segmentation, Point(cue_circle[0], cue_circle[1]), cue_circle[2], Scalar(segmentation_label::cue), FILLED);

    Vec3f black_circle = blls_localization.black.circle;
    circle(segmentation, Point(black_circle[0], black_circle[1]), black_circle[2], Scalar(segmentation_label::black), FILLED);

    for (ball_localization loc : blls_localization.solids)
    {
        Vec3f loc_circle = loc.circle;
        circle(segmentation, Point(loc_circle[0], loc_circle[1]), loc_circle[2], Scalar(segmentation_label::solids), FILLED);  
    }

    for (ball_localization loc : blls_localization.stripes)
    {
        Vec3f loc_circle = loc.circle;
        circle(segmentation, Point(loc_circle[0], loc_circle[1]), loc_circle[2], Scalar(segmentation_label::stripes), FILLED);  
    }

    dst = segmentation;
}