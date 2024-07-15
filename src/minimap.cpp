
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

void minimap::draw_dashed_line(Mat &img, Point pt1, Point pt2, Scalar color, int thickness, string style, int gap)
{
	float dx = pt1.x - pt2.x;
	float dy = pt1.y - pt2.y;
	float dist = hypot(dx, dy);

	vector<Point> to_draw;
	for (int i = 0; i < dist; i += gap)
	{
		float r = static_cast<float>(i / dist);
		int x = static_cast<int>((pt1.x * (1.0 - r) + pt2.x * r) + .5);
		int y = static_cast<int>((pt1.y * (1.0 - r) + pt2.y * r) + .5);
		to_draw.push_back(Point{x, y});
	}

	if (style == "dotted")
	{
		for (int i = 0; i < to_draw.size(); i++)
		{
			circle(img, to_draw.at(i), thickness, color, -1);
		}
	}
	else
	{
		Point start = to_draw.at(0);
		Point end = to_draw.at(0);

		for (int i = 0; i < to_draw.size(); i++)
		{
			start = end;
			end = to_draw.at(i);
			if (i % 2 == 1)
				line(img, start, end, color, thickness);
		}
	}
}

void minimap::get_balls_centers(const vector<Rect2d> &bounding_boxes, vector<Point> &balls_pos)
{
	for (Rect2d bounding_box : bounding_boxes)
	{
		balls_pos.push_back(Point2f(bounding_box.x + bounding_box.width / 2, bounding_box.y + bounding_box.height / 2));
	}
}

void minimap::draw_initial_minimap(const vector<Point> &corners, const vector<Point> &balls_pos, const Mat &src, Mat &dst)
{
	vector<Point2f> corners_minimap = {Point2f(70, 60), Point2f(924, 60), Point2f(924, 500), Point2f(70, 500)};
	vector<Point2f> corners_2f(corners.size());

	transform(corners.begin(), corners.end(), corners_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	Mat H = getPerspectiveTransform(corners_2f, corners_minimap);

	vector<Point2f> balls_pos_2f(balls_pos.size());
	transform(balls_pos.begin(), balls_pos.end(), balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	// Fill balls positions in minimap
	vector<Point2f> balls_pos_minimap;
	for (Point2f ball_pos : balls_pos_2f)
	{
		// Projected coordinate input and output arrays
		vector<Point2f> ball_pos_dst;
		vector<Point2f> ball_pos_src = {ball_pos};

		perspectiveTransform(ball_pos_src, ball_pos_dst, H);
		circle(dst, ball_pos_dst.at(0), 20, Scalar(0, 0, 255), FILLED);
		balls_pos_minimap.push_back(ball_pos_dst.at(0));
	}
}

void minimap::draw_minimap(const vector<Point> &corners, const vector<Point> &old_balls_pos, const vector<Point> &balls_pos, const Mat &src, Mat &trajectories, Mat &dst)
{
	vector<Point2f> corners_minimap = {Point2f(70, 60), Point2f(924, 60), Point2f(924, 500), Point2f(70, 500)};
	vector<Point2f> corners_2f(corners.size());
	transform(corners.begin(), corners.end(), corners_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });
	
	Mat H = getPerspectiveTransform(corners_2f, corners_minimap);

	vector<Point2f> balls_pos_2f(balls_pos.size());
	transform(balls_pos.begin(), balls_pos.end(), balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	vector<Point2f> old_balls_pos_2f(old_balls_pos.size());
	transform(old_balls_pos.begin(), old_balls_pos.end(), old_balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	// Fill balls positions in minimap
	vector<Point2f> balls_pos_minimap;
	for (int i = 0; i < balls_pos_2f.size(); i++)
	{
		// Projected coordinate input and output arrays
		vector<Point2f> ball_pos_dst;
		vector<Point2f> ball_pos_src = {balls_pos.at(i)};

		perspectiveTransform(ball_pos_src, ball_pos_dst, H);

		// Drawing trajectories for balls that moved more than DELTA_MOVEMENT
		const float DELTA_MOVEMENT = 5;
		if (norm(ball_pos_dst.at(0) - old_balls_pos_2f.at(i)) > DELTA_MOVEMENT)
			draw_dashed_line(trajectories, old_balls_pos_2f.at(i), ball_pos_dst.at(0), Scalar(0, 0, 0), 2, "dotted", 10);

		balls_pos_minimap.push_back(ball_pos_dst.at(0));
	}

	dst = trajectories.clone();
	for (Point2f ball_pos : balls_pos_minimap)
	{
		circle(dst, ball_pos, 20, Scalar(0, 0, 255), FILLED);
	}
}