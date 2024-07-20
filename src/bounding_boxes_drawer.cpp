#include "bounding_boxes_drawer.h"

using namespace std;
using namespace cv;

bounding_boxes_drawer::bounding_boxes_drawer(const playing_field_localization &plf_localization, const balls_localization &blls_localization, const std::vector<cv::Rect2d> &tracker_bboxes)
    : playing_field{plf_localization}, balls{blls_localization}
{
    vector<Point> balls_pos;
    for (Rect2d bounding_box : tracker_bboxes)
        balls_pos.push_back(Point(bounding_box.x + bounding_box.width / 2, bounding_box.y + bounding_box.height / 2));

    // Indeces associated to the solid balls in the multitracker.
    for (int i = 0; i < balls.solids.size(); i++)
    {
        vector<Point>::iterator it_solids;
        int x = balls.solids.at(i).bounding_box.x + balls.solids.at(i).bounding_box.width / 2;
        int y = balls.solids.at(i).bounding_box.y + balls.solids.at(i).bounding_box.height / 2;
        Point solid_ball_pos = Point(x, y);
        it_solids = find(balls_pos.begin(), balls_pos.end(), solid_ball_pos);
        solids_indeces.push_back(distance(balls_pos.begin(), it_solids));
    }

    // Indeces associated to the stripe balls in the multitracker.
    for (int i = 0; i < balls.stripes.size(); i++)
    {
        vector<Point>::iterator it_stripes;
        int x = balls.stripes.at(i).bounding_box.x + balls.stripes.at(i).bounding_box.width / 2;
        int y = balls.stripes.at(i).bounding_box.y + balls.stripes.at(i).bounding_box.height / 2;
        Point stripe_ball_pos = Point(x, y);
        it_stripes = find(balls_pos.begin(), balls_pos.end(), stripe_ball_pos);
        stripes_indeces.push_back(distance(balls_pos.begin(), it_stripes));
    }

    // Index associated to the black ball in the multitracker.
    vector<Point>::iterator it_black;
    int x_black = balls.black.bounding_box.x + balls.black.bounding_box.width / 2;
    int y_black = balls.black.bounding_box.y + balls.black.bounding_box.height / 2;
    Point black_ball_pos = Point(x_black, y_black);
    it_black = find(balls_pos.begin(), balls_pos.end(), black_ball_pos);
    black_index = distance(balls_pos.begin(), it_black);

    // Index associated to the cue ball in the multitracker.
    vector<Point>::iterator it_cue;
    int x_cue = balls.cue.bounding_box.x + balls.cue.bounding_box.width / 2;
    int y_cue = balls.cue.bounding_box.y + balls.cue.bounding_box.height / 2;
    Point cue_ball_pos = Point(x_cue, y_cue);
    it_cue = find(balls_pos.begin(), balls_pos.end(), cue_ball_pos);
    cue_index = distance(balls_pos.begin(), it_cue);
}

void bounding_boxes_drawer::draw(const cv::Mat &frame, cv::Mat &dst, const std::vector<cv::Rect2d> &updated_balls_bboxes)
{
    dst = frame.clone();

    const float ALPHA = 0.4;
    const Scalar WHITE = Scalar(255, 255, 255);
    const Scalar BLACK = Scalar(0, 0, 0);
    const Scalar BLUE = Scalar(255, 0, 0);
    const Scalar RED = Scalar(0, 0, 255);

    draw_transparent_rect(dst, updated_balls_bboxes.at(cue_index), WHITE, ALPHA);
    draw_transparent_rect(dst, updated_balls_bboxes.at(black_index), BLACK, ALPHA);

    for (int index : solids_indeces)
    {
        draw_transparent_rect(dst, updated_balls_bboxes.at(index), BLUE, ALPHA);
    }

    for (int index : stripes_indeces)
    {
        draw_transparent_rect(dst, updated_balls_bboxes.at(index), RED, ALPHA);
    }

    // Draw yellow lines
    vector<Point> corners = playing_field.corners;
    for (size_t i = 0; i < corners.size(); i++)
    {
        const Scalar YELLOW_COLOR = Scalar(0, 255, 255);
        const int LINE_THICKNESS = 3;
        line(dst, corners[i], corners[(i + 1) % corners.size()], YELLOW_COLOR, LINE_THICKNESS);
    }
}

void bounding_boxes_drawer::draw_transparent_rect(Mat &image, Rect rect, Scalar color, double alpha)
{
    Mat overlay;
    image.copyTo(overlay);
    rectangle(overlay, rect, color, FILLED);
    addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image);

    const int THICKNESS = 2;
    rectangle(image, rect, color, THICKNESS);
}