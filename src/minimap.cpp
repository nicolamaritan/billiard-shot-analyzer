// Author: Francesco Boscolo Meneguolo 2119969

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


minimap::minimap(playing_field_localization playing_field, balls_localization balls)
{
	corners_2f.resize(playing_field.corners.size());
	vector<Point> sorted_corners;
	sort_corners_for_minimap(playing_field.corners, sorted_corners);
	transform(sorted_corners.begin(), sorted_corners.end(), corners_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });
	projection_matrix = getPerspectiveTransform(corners_2f, corners_minimap);
}

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

bool minimap::is_rectangular_pool_table(const vector<Point> &pool_corners)
{
	const double EPS = 0.1;
    // Calculate the squared lengths of the four sides
    double d1 = norm(pool_corners[0] - pool_corners[1]);
    double d2 = norm(pool_corners[1] - pool_corners[2]);
    double d3 = norm(pool_corners[2] - pool_corners[3]);
    double d4 = norm(pool_corners[3] - pool_corners[0]);

    // Calculate the squared lengths of the two diagonals
    double diag1 = norm(pool_corners[0] - pool_corners[2]);
    double diag2 = norm(pool_corners[1] - pool_corners[3]);

    // Check if opposite sides are equal and diagonals are equal
    bool sides_equal = (abs(d1-d3) <= EPS) && (abs(d2 - d4) <= EPS);
    bool diagonals_equal = abs(diag1 - diag2) <= EPS;

    return sides_equal && diagonals_equal;
}


void minimap::sort_corners_for_minimap(const vector<Point> &corners_src, vector<Point> &corners_dst)
{
	int min_index = 0;
	const int EPS = 1;
	
	if(is_rectangular_pool_table(corners_src))
	{
		for (int i = 1; i < corners_src.size(); i++)
		{
			if(corners_src.at(i).y > corners_src.at(min_index).y)
				min_index = i;
			else if(abs(corners_src.at(i).y - corners_src.at(min_index).y) <= EPS)
			{
				if(corners_src.at(i).x < corners_src.at(min_index).x)
					min_index = i;
			}
		}

	}
	else
	{
		for (int i = 1; i < corners_src.size(); i++)
		{
			if(corners_src.at(i).y < corners_src.at(min_index).y)
				min_index = i;
			else if(abs(corners_src.at(i).y - corners_src.at(min_index).y) <= EPS)
			{
				if(corners_src.at(i).x < corners_src.at(min_index).x)
					min_index = i;
			}
		}
	}

	for (int i = 0; i < corners_src.size(); i++)
	{
		corners_dst.push_back(corners_src.at((min_index+i)%corners_src.size()));
	}

}



void minimap::get_balls_pos(const vector<Rect2d> &bounding_boxes, vector<Point> &balls_pos)
{
	for (Rect2d bounding_box : bounding_boxes)
	{
		balls_pos.push_back(Point(bounding_box.x + bounding_box.width / 2, bounding_box.y + bounding_box.height / 2));
	}
}

bool minimap::is_inside_hole(const cv::Point2f ball_position)
{
	bool inside_upper_left_hole = pow(ball_position.x - UPPER_LEFT_HOLE.x, 2) + pow(ball_position.y - UPPER_LEFT_HOLE.y, 2) <= pow(HOLE_RADIUS, 2);
	bool inside_upper_middle_hole = pow(ball_position.x - UPPER_MIDDLE_HOLE.x, 2) + pow(ball_position.y - UPPER_MIDDLE_HOLE.y, 2) <= pow(HOLE_RADIUS, 2);
	bool inside_upper_right_hole = pow(ball_position.x - UPPER_RIGHT_HOLE.x, 2) + pow(ball_position.y - UPPER_RIGHT_HOLE.y, 2) <= pow(HOLE_RADIUS, 2);
	bool inside_bottom_left_hole = pow(ball_position.x - BOTTOM_LEFT_HOLE.x, 2) + pow(ball_position.y - BOTTOM_LEFT_HOLE.y, 2) <= pow(HOLE_RADIUS, 2);
	bool inside_bottom_middle_hole = pow(ball_position.x - BOTTOM_MIDDLE_HOLE.x, 2) + pow(ball_position.y - BOTTOM_MIDDLE_HOLE.y, 2) <= pow(HOLE_RADIUS, 2);
	bool inside_bottom_right_hole = pow(ball_position.x - BOTTOM_RIGHT_HOLE.x, 2) + pow(ball_position.y - BOTTOM_RIGHT_HOLE.y, 2) <= pow(HOLE_RADIUS, 2);

	return inside_upper_left_hole || inside_upper_middle_hole || inside_upper_right_hole || inside_bottom_left_hole || inside_bottom_middle_hole || inside_bottom_right_hole;
}


bool minimap::is_inside_playing_field(const Point2f ball_position)
{
	const int EPS = 4;
	return !is_inside_hole(ball_position) && (ball_position.x > X1+EPS && ball_position.x < X2-EPS) && (ball_position.y > Y1+EPS && ball_position.y < Y2-EPS);
}



void minimap::draw_initial_minimap(const vector<Point> &balls_pos, const balls_localization &balls, vector<int> &solids_indeces, vector<int> &stripes_indeces, int &black_index, int &cue_index, const Mat &src, Mat &dst)
{
	vector<Point> copy_balls_pos(balls_pos.size());
	copy(balls_pos.begin(), balls_pos.end(), copy_balls_pos.begin());
	for(int i = 0; i < balls.solids.size(); i++)
	{
		vector<Point>::iterator it_solids;
		int x = balls.solids.at(i).bounding_box.x + balls.solids.at(i).bounding_box.width/2;
		int y = balls.solids.at(i).bounding_box.y + balls.solids.at(i).bounding_box.height/2;
		Point solid_ball_pos = Point(x, y);
		it_solids = find(copy_balls_pos.begin(), copy_balls_pos.end(), solid_ball_pos);
		solids_indeces.push_back(distance(copy_balls_pos.begin(), it_solids));
	}
	for(int i = 0; i < balls.stripes.size(); i++)
	{
		vector<Point>::iterator it_stripes;
		int x = balls.stripes.at(i).bounding_box.x + balls.stripes.at(i).bounding_box.width/2;
		int y = balls.stripes.at(i).bounding_box.y + balls.stripes.at(i).bounding_box.height/2;
		Point stripe_ball_pos = Point(x, y);
		it_stripes = find(copy_balls_pos.begin(), copy_balls_pos.end(), stripe_ball_pos);
		stripes_indeces.push_back(distance(copy_balls_pos.begin(), it_stripes));
	}


	vector<Point>::iterator it_black;
	int x_black = balls.black.bounding_box.x + balls.black.bounding_box.width/2;
	int y_black = balls.black.bounding_box.y + balls.black.bounding_box.height/2;
	Point black_ball_pos = Point(x_black, y_black);
	it_black = find(copy_balls_pos.begin(), copy_balls_pos.end(), black_ball_pos);
	black_index = distance(copy_balls_pos.begin(), it_black);


	vector<Point>::iterator it_cue;
	int x_cue = balls.cue.bounding_box.x + balls.cue.bounding_box.width/2;
	int y_cue = balls.cue.bounding_box.y + balls.cue.bounding_box.height/2;
	Point cue_ball_pos = Point(x_cue, y_cue);
	it_cue = find(copy_balls_pos.begin(), copy_balls_pos.end(), cue_ball_pos);
	cue_index = distance(copy_balls_pos.begin(), it_cue);


	vector<Point2f> balls_pos_2f(balls_pos.size());
	transform(balls_pos.begin(), balls_pos.end(), balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	
	for(int i = 0; i < solids_indeces.size(); i++)
	{
		// Projected coordinate input and output arrays
		vector<Point2f> ball_pos_dst;
		vector<Point2f> ball_pos_src = {balls_pos_2f.at(solids_indeces.at(i))}; 
		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix);
		circle(dst, ball_pos_dst.at(0), BALL_RADIUS, SOLID_BALL_COLOR, FILLED);
		circle(dst, ball_pos_dst.at(0), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
	}

	for(int i = 0; i < stripes_indeces.size(); i++)
	{
		// Projected coordinate input and output arrays
		vector<Point2f> ball_pos_dst;
		vector<Point2f> ball_pos_src = {balls_pos_2f.at(stripes_indeces.at(i))};
		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix);
		circle(dst, ball_pos_dst.at(0), BALL_RADIUS, STRIPE_BALL_COLOR, FILLED);
		circle(dst, ball_pos_dst.at(0), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
	}
	

	vector<Point2f> black_ball_pos_dst;
	vector<Point2f> black_ball_pos_src = {balls_pos_2f.at(black_index)};
	perspectiveTransform(black_ball_pos_src, black_ball_pos_dst, projection_matrix);
	circle(dst, black_ball_pos_dst.at(0), BALL_RADIUS, BLACK_BALL_COLOR, FILLED);
	circle(dst, black_ball_pos_dst.at(0), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
	

	vector<Point2f> cue_ball_pos_dst;
	vector<Point2f> cue_ball_pos_src = {balls_pos_2f.at(cue_index)};
	perspectiveTransform(cue_ball_pos_src, cue_ball_pos_dst, projection_matrix);
	circle(dst, cue_ball_pos_dst.at(0), BALL_RADIUS, CUE_BALL_COLOR, FILLED);
	circle(dst, cue_ball_pos_dst.at(0), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);

}



void minimap::draw_minimap(const vector<Point> &old_balls_pos, const vector<Point> &balls_pos, const std::vector<int> &solids_indeces, const std::vector<int> &stripes_indeces, const int black_index, const int cue_index, const Mat &src, Mat &trajectories, Mat &dst)
{
	const float DELTA_MOVEMENT = 2;
	// Fill balls positions in minimap
	vector<Point2f> solids_balls_pos_minimap;
	vector<Point2f> stripes_balls_pos_minimap;
	Point2f black_ball_pos_minimap;
	Point2f cue_ball_pos_minimap;


	vector<Point2f> balls_pos_2f(balls_pos.size());
	transform(balls_pos.begin(), balls_pos.end(), balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	vector<Point2f> old_balls_pos_2f(old_balls_pos.size());
	transform(old_balls_pos.begin(), old_balls_pos.end(), old_balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });


	for(int i = 0; i < solids_indeces.size(); i++)
	{
		vector<Point2f> ball_pos_dst;
		vector<Point2f> ball_pos_src = {balls_pos_2f.at(solids_indeces.at(i))};

		vector<Point2f> old_ball_pos_dst;
		vector<Point2f> old_ball_pos_src = {old_balls_pos_2f.at(solids_indeces.at(i))};

		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix);
		perspectiveTransform(old_ball_pos_src, old_ball_pos_dst, projection_matrix);

		solids_balls_pos_minimap.push_back(ball_pos_dst.at(0));

		// Drawing trajectories for balls that moved more than DELTA_MOVEMENT
		if(balls_pos_2f.at(solids_indeces.at(i)) != INVALID_POSITION && norm(ball_pos_dst.at(0) - old_ball_pos_dst.at(0)) > DELTA_MOVEMENT && is_inside_playing_field(solids_balls_pos_minimap.at(i)))
			draw_dashed_line(trajectories, old_ball_pos_dst.at(0), ball_pos_dst.at(0), CONTOUR_COLOR, THICKNESS, "dotted", GAP);
	}

	for(int i = 0; i < stripes_indeces.size(); i++)
	{
		vector<Point2f> ball_pos_dst;
		vector<Point2f> ball_pos_src = {balls_pos_2f.at(stripes_indeces.at(i))};

		vector<Point2f> old_ball_pos_dst;
		vector<Point2f> old_ball_pos_src = {old_balls_pos_2f.at(stripes_indeces.at(i))};

		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix);
		perspectiveTransform(old_ball_pos_src, old_ball_pos_dst, projection_matrix);

		stripes_balls_pos_minimap.push_back(ball_pos_dst.at(0));

		if(balls_pos_2f.at(stripes_indeces.at(i)) != INVALID_POSITION && norm(ball_pos_dst.at(0) - old_ball_pos_dst.at(0)) > DELTA_MOVEMENT && is_inside_playing_field(stripes_balls_pos_minimap.at(i)))
		{
			// Drawing trajectories for balls that moved more than DELTA_MOVEMENT
			draw_dashed_line(trajectories, old_ball_pos_dst.at(0), ball_pos_dst.at(0), CONTOUR_COLOR, THICKNESS, "dotted", GAP);
		}
	}
	

	vector<Point2f> black_ball_pos_dst;
	vector<Point2f> black_ball_pos_src = {balls_pos_2f.at(black_index)};
	vector<Point2f> old_black_ball_pos_dst;
	vector<Point2f> old_black_ball_pos_src = {old_balls_pos_2f.at(black_index)};
	perspectiveTransform(black_ball_pos_src, black_ball_pos_dst, projection_matrix);
	perspectiveTransform(old_black_ball_pos_src, old_black_ball_pos_dst, projection_matrix);
	black_ball_pos_minimap = black_ball_pos_dst.at(0);
	if(balls_pos_2f.at(black_index) != INVALID_POSITION)
	{
		if(norm(black_ball_pos_dst.at(0) - old_black_ball_pos_dst.at(0)) > DELTA_MOVEMENT && (is_inside_playing_field(black_ball_pos_minimap) && !is_inside_hole(black_ball_pos_minimap)))
			draw_dashed_line(trajectories, old_black_ball_pos_dst.at(0), black_ball_pos_dst.at(0), CONTOUR_COLOR, THICKNESS, "dotted", GAP);
	}


	vector<Point2f> cue_ball_pos_dst;
	vector<Point2f> cue_ball_pos_src = {balls_pos_2f.at(cue_index)};
	vector<Point2f> old_cue_ball_pos_dst;
	vector<Point2f> old_cue_ball_pos_src = {old_balls_pos_2f.at(cue_index)};
	perspectiveTransform(cue_ball_pos_src, cue_ball_pos_dst, projection_matrix);
	perspectiveTransform(old_cue_ball_pos_src, old_cue_ball_pos_dst, projection_matrix);
	if (norm(cue_ball_pos_dst.at(0) - old_cue_ball_pos_dst.at(0)) > DELTA_MOVEMENT)
		draw_dashed_line(trajectories, old_cue_ball_pos_dst.at(0), cue_ball_pos_dst.at(0), CONTOUR_COLOR, THICKNESS, "dotted", GAP);

	cue_ball_pos_minimap = cue_ball_pos_dst.at(0);

	
	dst = trajectories.clone();

	// Draw solid balls
	for(int i = 0; i < solids_indeces.size(); i++)
	{
		if(is_inside_playing_field(solids_balls_pos_minimap.at(i)))
		{
			circle(dst, solids_balls_pos_minimap.at(i), BALL_RADIUS, SOLID_BALL_COLOR, FILLED);
			circle(dst, solids_balls_pos_minimap.at(i), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
		}
	}

	// Draw stripe  balls
	for(int i = 0; i < stripes_indeces.size(); i++)
	{
		if(is_inside_playing_field(stripes_balls_pos_minimap.at(i)))
		{
			circle(dst, stripes_balls_pos_minimap.at(i), BALL_RADIUS, STRIPE_BALL_COLOR, FILLED);
			circle(dst, stripes_balls_pos_minimap.at(i), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
		}
	}

	// Draw black ball
	if(is_inside_playing_field(black_ball_pos_minimap))
	{
		circle(dst, black_ball_pos_minimap, BALL_RADIUS, BLACK_BALL_COLOR, FILLED);
		circle(dst, black_ball_pos_minimap, BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
	}

	// Draw cue ball
	circle(dst, cue_ball_pos_minimap, BALL_RADIUS, CUE_BALL_COLOR, FILLED);
	circle(dst, cue_ball_pos_minimap, BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
	

}



/*
void minimap::draw_initial_minimap(const vector<Point> &balls_pos, const Mat &src, Mat &dst)
{

	vector<Point2f> balls_pos_2f(balls_pos.size());
	transform(balls_pos.begin(), balls_pos.end(), balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	// Fill balls positions in minimap
	//vector<Point2f> balls_pos_minimap;
	for (Point2f ball_pos : balls_pos_2f)
	{
		// Projected coordinate input and output arrays
		vector<Point2f> ball_pos_dst;
		vector<Point2f> ball_pos_src = {ball_pos};

		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix);
		circle(dst, ball_pos_dst.at(0), 20, Scalar(0, 0, 255), FILLED);
		//balls_pos_minimap.push_back(ball_pos_dst.at(0));
	}
}
*/
/*
void minimap::draw_minimap(const vector<Point> &old_balls_pos, const vector<Point> &balls_pos, const Mat &src, Mat &trajectories, Mat &dst)
{
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
		vector<Point2f> ball_pos_src = {balls_pos_2f.at(i)};

		vector<Point2f> old_ball_pos_dst;
		vector<Point2f> old_ball_pos_src = {old_balls_pos_2f.at(i)};

		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix);
		perspectiveTransform(old_ball_pos_src, old_ball_pos_dst, projection_matrix);

		// Drawing trajectories for balls that moved more than DELTA_MOVEMENT
		const float DELTA_MOVEMENT = 2;
		if (norm(ball_pos_dst.at(0) - old_ball_pos_dst.at(0)) > DELTA_MOVEMENT)
			draw_dashed_line(trajectories, old_ball_pos_dst.at(0), ball_pos_dst.at(0), Scalar(0, 0, 0), 2, "dotted", 10);

		balls_pos_minimap.push_back(ball_pos_dst.at(0));
	}

	dst = trajectories.clone();
	for (Point2f ball_pos : balls_pos_minimap)
	{
		circle(dst, ball_pos, 10, Scalar(0, 0, 255), FILLED);
	}
}

*/