
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "minimap.h"


#include <iostream>
#include <cmath>
#include <map>
#include <queue>
#include <cassert>

using namespace cv;
using namespace std;

void minimap::draw_dashed_line(cv::Mat& img, cv::Point pt1, cv::Point pt2, cv::Scalar color, int thickness, std::string style, int gap)
{
  float dx = pt1.x - pt2.x;
  float dy = pt1.y - pt2.y;
  float dist = std::hypot(dx, dy);

  std::vector<cv::Point> pts;
  for (int i = 0; i < dist; i += gap)
  {
    float r = static_cast<float>(i / dist);
    int x = static_cast<int>((pt1.x * (1.0 - r) + pt2.x * r) + .5);
    int y = static_cast<int>((pt1.y * (1.0 - r) + pt2.y * r) + .5);
    pts.emplace_back(x, y);
  }

  int pts_size = pts.size();

  if (style == "dotted")
  {
    for (int i = 0; i < pts_size; ++i)
    {
      cv::circle(img, pts[i], thickness, color, -1);
    }
  } 
  else
  {
    cv::Point s = pts[0];
    cv::Point e = pts[0];

    for (int i = 0; i < pts_size; ++i)
    {
      s = e;
      e = pts[i];
      if (i % 2 == 1)
      {
        cv::line(img, s, e, color, thickness);
      }
    }
  }
}


vector<Point> minimap::get_balls_pos(vector<Rect2d> bounding_boxes)
{
    vector<Point> balls_pos;
    for(const Rect2d bounding_box : bounding_boxes)
    {
        balls_pos.push_back(Point2f(bounding_box.x + bounding_box.width/2, bounding_box.y + bounding_box.height/2));
    }
    return balls_pos;
}

void minimap::draw_initial_minimap(const vector<Point> corners_src, const vector<Point> &balls_src, const Mat &src, Mat &dst)
{
    vector<Point> corners_dst = {Point(70, 60), Point(924, 60), Point(924, 500), Point(70, 500)};
    Mat H = getPerspectiveTransform(corners_src, corners_dst);
    vector<Point2f> balls_pos_dst;
    int i = 0;
    for (const Point &ball : balls_src)
    {
        vector<Point> ball_pos_dst;

        vector<Point> ball_pos_src = {ball};
        perspectiveTransform(ball_pos_src, ball_pos_dst, H);
        circle(dst, ball_pos_dst[0], 20, Scalar(0, 0, 255), FILLED);
        balls_pos_dst.push_back(ball_pos_dst[0]);
    }

}

void minimap::draw_minimap(const std::vector<cv::Point> &corners_src, const std::vector<cv::Point> &old_balls_pos, const std::vector<cv::Point> &balls_src, const cv::Mat &src, cv::Mat &trajectories, cv::Mat &dst)
{
    vector<Point> corners_dst = {Point(70, 60), Point(924, 60), Point(924, 500), Point(70, 500)};
    Mat H = getPerspectiveTransform(corners_src, corners_dst);
    vector<Point> balls_pos_dst;
    int i = 0;
    for (const Point &ball : balls_src)
    {
        vector<Point> ball_pos_dst;

        vector<Point> ball_pos_src = {ball};
        perspectiveTransform(ball_pos_src, ball_pos_dst, H);
        if(norm(ball_pos_dst[0] - old_balls_pos[i]) > 5)
        {
            draw_dashed_line(trajectories, old_balls_pos[i], ball_pos_dst[0], Scalar(0, 0, 0),
                            2, "dotted", 10);
        }
        
        balls_pos_dst.push_back(ball_pos_dst[0]);
    }

    dst = trajectories.clone();
    for(const Point2f &ball : balls_pos_dst)
    {
        circle(dst, ball, 20, Scalar(0, 0, 255), FILLED);
    }

    //return balls_pos_dst;
}