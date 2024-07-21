// Author: Francesco Boscolo Meneguolo 2119969

#include "minimap.h"
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

minimap::minimap(const playing_field_localization &plf_localization, const balls_localization &blls_localization, const std::vector<cv::Rect2d> &tracker_bboxes)
	: playing_field{plf_localization}, balls{blls_localization}
{
	const int BLACK_AND_CUE_BALLS_NUMBER = 2;
	if (tracker_bboxes.size() != (blls_localization.solids.size() + blls_localization.stripes.size() + BLACK_AND_CUE_BALLS_NUMBER))
	{
		const string INVALID_BBOXES_SIZE = "Tracker bounding boxes and localization bounding boxes do not match in size.";
		throw invalid_argument(INVALID_BBOXES_SIZE);
	}

	get_balls_pos(tracker_bboxes, current_balls_pos);
	load_balls_indeces(current_balls_pos);

	corners_2f.resize(playing_field.corners.size());
	vector<Point> sorted_corners;
	sort_corners_for_minimap(playing_field.corners, sorted_corners);
	transform(sorted_corners.begin(), sorted_corners.end(), corners_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });
	projection_matrix = getPerspectiveTransform(corners_2f, corners_minimap); // Projection matrix from points in the input frame to the minimap.

	fs::path MINIMAP_PATH = fs::path(IMAGES_DIRECTORY) / fs::path(MINIMAP_IMAGE_FILENAME);
	empty_minimap = imread(MINIMAP_PATH.string());
	trajectories = empty_minimap.clone();
};

void minimap::draw_initial_minimap(Mat &dst)
{
	dst = empty_minimap.clone();

	vector<Point2f> balls_pos_2f(current_balls_pos.size());
	transform(current_balls_pos.begin(), current_balls_pos.end(), balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	for (int i = 0; i < solids_indeces.size(); i++)
	{
		vector<Point2f> ball_pos_dst; // Projected coordinates of a solid ball.
		vector<Point2f> ball_pos_src = {balls_pos_2f.at(solids_indeces.at(i))}; // Input frame coordinates of a solid ball.
		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix);
		// Draw solid ball in the minimap.
		circle(dst, ball_pos_dst.at(0), BALL_RADIUS, SOLID_BALL_COLOR, FILLED);
		circle(dst, ball_pos_dst.at(0), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
	}

	for (int i = 0; i < stripes_indeces.size(); i++)
	{
		vector<Point2f> ball_pos_dst; // Projected coordinates of a stripe ball.
		vector<Point2f> ball_pos_src = {balls_pos_2f.at(stripes_indeces.at(i))}; // Input frame coordinates of a stripe ball.
		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix);
		// Draw stripe ball in the minimap.
		circle(dst, ball_pos_dst.at(0), BALL_RADIUS, STRIPE_BALL_COLOR, FILLED);
		circle(dst, ball_pos_dst.at(0), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
	}

	
	vector<Point2f> black_ball_pos_dst; // Projected coordinates of the black ball.
	vector<Point2f> black_ball_pos_src = {balls_pos_2f.at(black_index)}; // Input frame coordinates of the black ball.
	perspectiveTransform(black_ball_pos_src, black_ball_pos_dst, projection_matrix);
	// Draw black ball in the minimap.
	circle(dst, black_ball_pos_dst.at(0), BALL_RADIUS, BLACK_BALL_COLOR, FILLED);
	circle(dst, black_ball_pos_dst.at(0), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);

	vector<Point2f> cue_ball_pos_dst; // Projected coordinates of the cue ball.
	vector<Point2f> cue_ball_pos_src = {balls_pos_2f.at(cue_index)}; // Input frame coordinates of the cue ball.
	perspectiveTransform(cue_ball_pos_src, cue_ball_pos_dst, projection_matrix);
	// Draw cue ball in the minimap.
	circle(dst, cue_ball_pos_dst.at(0), BALL_RADIUS, CUE_BALL_COLOR, FILLED);
	circle(dst, cue_ball_pos_dst.at(0), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
}

void minimap::draw_minimap(Mat &dst)
{
	dst = trajectories.clone();

	const float DELTA_MOVEMENT = 2;
	// Fill balls positions in minimap
	vector<Point2f> solids_balls_pos_minimap;
	vector<Point2f> stripes_balls_pos_minimap;
	Point2f black_ball_pos_minimap;
	Point2f cue_ball_pos_minimap;

	vector<Point2f> current_balls_pos_2f(current_balls_pos.size());
	transform(current_balls_pos.begin(), current_balls_pos.end(), current_balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	vector<Point2f> old_balls_pos_2f(old_balls_pos.size());
	transform(old_balls_pos.begin(), old_balls_pos.end(), old_balls_pos_2f.begin(), [](const Point &point)
			  { return static_cast<Point2f>(point); });

	for (int i = 0; i < solids_indeces.size(); i++)
	{
		int index = solids_indeces.at(i);

		vector<Point2f> ball_pos_dst; // Current projected coordinates of a solid ball.
		vector<Point2f> ball_pos_src = {current_balls_pos_2f.at(index)}; // Current input frame coordinates of a solid ball.

		vector<Point2f> old_ball_pos_dst; // Previous projected coordinates of a solid ball.
		vector<Point2f> old_ball_pos_src = {old_balls_pos_2f.at(index)}; // Previous input frame coordinates of a solid ball.

		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix); // Current perspective transform.
		perspectiveTransform(old_ball_pos_src, old_ball_pos_dst, projection_matrix); // Previous perspective transform.

		solids_balls_pos_minimap.push_back(ball_pos_dst.at(0)); // Solid balls positions in the minimap.

		// Drawing trajectories for balls that moved more than DELTA_MOVEMENT.
		if(current_balls_pos_2f.at(index) != INVALID_POSITION && norm(ball_pos_dst.at(0) - old_ball_pos_dst.at(0)) > DELTA_MOVEMENT && is_inside_playing_field(solids_balls_pos_minimap.at(i)))
			draw_dotted_line(trajectories, old_ball_pos_dst.at(0), ball_pos_dst.at(0), CONTOUR_COLOR, THICKNESS, GAP);
	}

	for (int i = 0; i < stripes_indeces.size(); i++)
	{
		int index = stripes_indeces.at(i);

		vector<Point2f> ball_pos_dst; // Current projected coordinates of a stripe ball.
		vector<Point2f> ball_pos_src = {current_balls_pos_2f.at(index)}; // Current input frame coordinates of a stripe ball.

		vector<Point2f> old_ball_pos_dst; // Previous projected coordinates of a stripe ball.
		vector<Point2f> old_ball_pos_src = {old_balls_pos_2f.at(index)}; // Previous input frame coordinates of a stripe ball.

		perspectiveTransform(ball_pos_src, ball_pos_dst, projection_matrix); // Current perspective transform.
		perspectiveTransform(old_ball_pos_src, old_ball_pos_dst, projection_matrix); // Previous perspective transform.

		stripes_balls_pos_minimap.push_back(ball_pos_dst.at(0)); // Stripe balls positions in the minimap.

		// Drawing trajectories for balls that moved more than DELTA_MOVEMENT.
		if (current_balls_pos_2f.at(index) != INVALID_POSITION && norm(ball_pos_dst.at(0) - old_ball_pos_dst.at(0)) > DELTA_MOVEMENT && (is_inside_playing_field(stripes_balls_pos_minimap.at(i))))
			draw_dotted_line(trajectories, old_ball_pos_dst.at(0), ball_pos_dst.at(0), CONTOUR_COLOR, THICKNESS, GAP);
	}

	vector<Point2f> black_ball_pos_dst; // Current projected coordinates of the black ball.
	vector<Point2f> black_ball_pos_src = {current_balls_pos_2f.at(black_index)}; // Current input frame coordinates of the black ball.
	vector<Point2f> old_black_ball_pos_dst; // Previous projected coordinates of the black ball.
	vector<Point2f> old_black_ball_pos_src = {old_balls_pos_2f.at(black_index)}; // Previous input frame coordinates of the black ball.

	perspectiveTransform(black_ball_pos_src, black_ball_pos_dst, projection_matrix); // Current perspective transform.
	perspectiveTransform(old_black_ball_pos_src, old_black_ball_pos_dst, projection_matrix); // Previous perspective transform.
	black_ball_pos_minimap = black_ball_pos_dst.at(0); // Black ball position in the minimap.

	// Drawing trajectory for black ball if it moved more than DELTA_MOVEMENT.
	if (current_balls_pos_2f.at(black_index) != INVALID_POSITION && norm(black_ball_pos_dst.at(0) - old_black_ball_pos_dst.at(0)) > DELTA_MOVEMENT && is_inside_playing_field(black_ball_pos_minimap))
		draw_dotted_line(trajectories, old_black_ball_pos_dst.at(0), black_ball_pos_dst.at(0), CONTOUR_COLOR, THICKNESS, GAP);

	vector<Point2f> cue_ball_pos_dst; // Current projected coordinates of the cue ball.
	vector<Point2f> cue_ball_pos_src = {current_balls_pos_2f.at(cue_index)}; // Current input frame coordinates of the cue ball.
	vector<Point2f> old_cue_ball_pos_dst; // Previous projected coordinates of the cue ball.
	vector<Point2f> old_cue_ball_pos_src = {old_balls_pos_2f.at(cue_index)}; // Previous input frame coordinates of the cue ball.
	perspectiveTransform(cue_ball_pos_src, cue_ball_pos_dst, projection_matrix); // Current perspective transform.
	perspectiveTransform(old_cue_ball_pos_src, old_cue_ball_pos_dst, projection_matrix); // Previous perspective transform.
	cue_ball_pos_minimap = cue_ball_pos_dst.at(0); // Cue ball position in the minimap.

	// Drawing trajectory for cue ball if it moved more than DELTA_MOVEMENT.
	if (current_balls_pos_2f.at(cue_index) != INVALID_POSITION && norm(cue_ball_pos_dst.at(0) - old_cue_ball_pos_dst.at(0)) > DELTA_MOVEMENT && is_inside_playing_field(cue_ball_pos_minimap))
		draw_dotted_line(trajectories, old_cue_ball_pos_dst.at(0), cue_ball_pos_dst.at(0), CONTOUR_COLOR, THICKNESS, GAP);

	// Start to build the minimap image from the image of the trajectories of the balls.
	dst = trajectories.clone();

	// Draw solid balls
	for (int i = 0; i < solids_indeces.size(); i++)
	{
		if (is_inside_playing_field(solids_balls_pos_minimap.at(i)))
		{
			circle(dst, solids_balls_pos_minimap.at(i), BALL_RADIUS, SOLID_BALL_COLOR, FILLED);
			circle(dst, solids_balls_pos_minimap.at(i), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
		}
	}

	// Draw stripe balls
	for (int i = 0; i < stripes_indeces.size(); i++)
	{
		if (is_inside_playing_field(stripes_balls_pos_minimap.at(i)))
		{
			circle(dst, stripes_balls_pos_minimap.at(i), BALL_RADIUS, STRIPE_BALL_COLOR, FILLED);
			circle(dst, stripes_balls_pos_minimap.at(i), BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
		}
	}

	// Draw black ball
	if (is_inside_playing_field(black_ball_pos_minimap))
	{
		circle(dst, black_ball_pos_minimap, BALL_RADIUS, BLACK_BALL_COLOR, FILLED);
		circle(dst, black_ball_pos_minimap, BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
	}

	// Draw cue ball
	circle(dst, cue_ball_pos_minimap, BALL_RADIUS, CUE_BALL_COLOR, FILLED);
	circle(dst, cue_ball_pos_minimap, BALL_RADIUS, CONTOUR_COLOR, THICKNESS);
}

void minimap::update(const std::vector<cv::Rect2d> &updated_balls_bboxes)
{
	if (updated_balls_bboxes.size() != current_balls_pos.size())
	{
		const string INVALID_BALLS_BBOXES = "Updated bounding boxes number does not match current bounding boxes number.";
		throw invalid_argument(INVALID_BALLS_BBOXES);
	}

	old_balls_pos.clear();
	copy(current_balls_pos.begin(), current_balls_pos.end(), back_inserter(old_balls_pos));
	get_balls_pos(updated_balls_bboxes, current_balls_pos);
}

void minimap::get_balls_pos(const vector<Rect2d> &bounding_boxes, vector<Point> &balls_pos)
{
	balls_pos.clear();
	for (Rect2d bounding_box : bounding_boxes)
		balls_pos.push_back(Point(bounding_box.x + bounding_box.width / 2, bounding_box.y + bounding_box.height / 2));
}

void minimap::load_balls_indeces(const vector<Point> &balls_pos)
{
	vector<Point> copy_balls_pos(balls_pos.size());
	copy(balls_pos.begin(), balls_pos.end(), copy_balls_pos.begin());

	//Indeces associated to the solid balls in the multitracker.
	for (int i = 0; i < balls.solids.size(); i++)
	{
		vector<Point>::iterator it_solids;
		int x = balls.solids.at(i).bounding_box.x + balls.solids.at(i).bounding_box.width / 2;
		int y = balls.solids.at(i).bounding_box.y + balls.solids.at(i).bounding_box.height / 2;
		Point solid_ball_pos = Point(x, y);
		it_solids = find(copy_balls_pos.begin(), copy_balls_pos.end(), solid_ball_pos);
		solids_indeces.push_back(distance(copy_balls_pos.begin(), it_solids));
	}

	//Indeces associated to the stripe balls in the multitracker.
	for (int i = 0; i < balls.stripes.size(); i++)
	{
		vector<Point>::iterator it_stripes;
		int x = balls.stripes.at(i).bounding_box.x + balls.stripes.at(i).bounding_box.width / 2;
		int y = balls.stripes.at(i).bounding_box.y + balls.stripes.at(i).bounding_box.height / 2;
		Point stripe_ball_pos = Point(x, y);
		it_stripes = find(copy_balls_pos.begin(), copy_balls_pos.end(), stripe_ball_pos);
		stripes_indeces.push_back(distance(copy_balls_pos.begin(), it_stripes));
	}

	// Index associated to the black ball in the multitracker.
	vector<Point>::iterator it_black;
	int x_black = balls.black.bounding_box.x + balls.black.bounding_box.width / 2;
	int y_black = balls.black.bounding_box.y + balls.black.bounding_box.height / 2;
	Point black_ball_pos = Point(x_black, y_black);
	it_black = find(copy_balls_pos.begin(), copy_balls_pos.end(), black_ball_pos);
	black_index = distance(copy_balls_pos.begin(), it_black);

	//Index associated to the cue ball in the multitracker.
	vector<Point>::iterator it_cue;
	int x_cue = balls.cue.bounding_box.x + balls.cue.bounding_box.width / 2;
	int y_cue = balls.cue.bounding_box.y + balls.cue.bounding_box.height / 2;
	Point cue_ball_pos = Point(x_cue, y_cue);
	it_cue = find(copy_balls_pos.begin(), copy_balls_pos.end(), cue_ball_pos);
	cue_index = distance(copy_balls_pos.begin(), it_cue);
}

bool minimap::is_rectangular_pool_table(const vector<Point> &pool_corners)
{
	const double EPS = 0.1;
	// Calculate the squared lengths of the four sides.
	double d1 = norm(pool_corners[0] - pool_corners[1]);
	double d2 = norm(pool_corners[1] - pool_corners[2]);
	double d3 = norm(pool_corners[2] - pool_corners[3]);
	double d4 = norm(pool_corners[3] - pool_corners[0]);

	// Calculate the squared lengths of the two diagonals.
	double diag1 = norm(pool_corners[0] - pool_corners[2]);
	double diag2 = norm(pool_corners[1] - pool_corners[3]);

	// Check if opposite sides are equal and diagonals are equal.
	bool sides_equal = (abs(d1 - d3) <= EPS) && (abs(d2 - d4) <= EPS);
	bool diagonals_equal = abs(diag1 - diag2) <= EPS;

	return sides_equal && diagonals_equal;
}

bool minimap::is_inside_playing_field(const Point2f ball_position)
{
	const int EPS = 4;
	return !is_inside_hole(ball_position) && (ball_position.x > X1 + EPS && ball_position.x < X2 - EPS) && (ball_position.y > Y1 + EPS && ball_position.y < Y2 - EPS);
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

void minimap::sort_corners_for_minimap(const vector<Point> &original_corners, vector<Point> &sorted_corners)
{
	int first_pos_index = 0;
	const int EPS = 1;

	/* 
		If the table in the input frame is a rectangle the corners of the table are sorted
		in order to mantain its orientation in the minimap,
		putting in first position the bottom left corner (considering the input frame)
		and the others follow in clockwise order.

	*/
	if (is_rectangular_pool_table(original_corners))
	{
		for (int i = 1; i < original_corners.size(); i++)
		{
			if (original_corners.at(i).y > original_corners.at(first_pos_index).y)
				first_pos_index = i;
			else if (abs(original_corners.at(i).y - original_corners.at(first_pos_index).y) <= EPS)
			{
				if (original_corners.at(i).x < original_corners.at(first_pos_index).x)
					first_pos_index = i;
			}
		}
	}
	/*
		If the table in the input frame is not a rectangle the corners are sorted
		in order to put in first position the corner that corresponds to
		the bottom left corner of the minimap table, i.e. the corner in the input frame
		with the lowest y-coordinate (and also the lowest x-coordinate in case of tie),
		and the others follow in clockwise order.
		We used the convention that the farthest side of the table from the camera in the input frame 
		corresponds to the left vertical side in the minimap.
	*/
	else
	{
		for (int i = 1; i < original_corners.size(); i++)
		{
			if (original_corners.at(i).y < original_corners.at(first_pos_index).y)
				first_pos_index = i;
			else if (abs(original_corners.at(i).y - original_corners.at(first_pos_index).y) <= EPS)
			{
				if (original_corners.at(i).x < original_corners.at(first_pos_index).x)
					first_pos_index = i;
			}
		}
	}

	// Sorted corners with criterions specified above.
	for (int i = 0; i < original_corners.size(); i++)
		sorted_corners.push_back(original_corners.at((first_pos_index + i) % original_corners.size()));
}

void minimap::draw_dotted_line(Mat &img, Point pt1, Point pt2, Scalar color, int thickness, int gap)
{
	float dx = pt1.x - pt2.x;
	float dy = pt1.y - pt2.y;
	float dist = hypot(dx, dy);

	// Points to draw.
	vector<Point> to_draw;
	for (int i = 0; i < dist; i += gap)
	{
		float r = static_cast<float>(i / dist);
		int x = static_cast<int>((pt1.x * (1.0 - r) + pt2.x * r) + .5);
		int y = static_cast<int>((pt1.y * (1.0 - r) + pt2.y * r) + .5);
		to_draw.push_back(Point{x, y});
	}

	// Draw dotted line.
	for (int i = 0; i < to_draw.size(); i++)
		circle(img, to_draw.at(i), thickness, color, FILLED);
}