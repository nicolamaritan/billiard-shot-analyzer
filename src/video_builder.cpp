#include "video_builder.h"
#include "minimap.h"
#include "playing_field_localization.h"
#include "balls_localization.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

#include <filesystem>

using namespace cv;
using namespace std;

void video_builder::build_video(vector<Mat> output_frames, string output_filename)
{
    VideoWriter output_video;
    output_video.open(output_filename, input_video_codec, input_video_fps, input_video_size, true);
    if (!output_video.isOpened())
    {
        cerr << "Could not open the output video for write." << endl;
        return;
    }
    for (Mat output_frame : output_frames)
    {
        output_video << output_frame;
    }
}

void video_builder::build_videos(string dataset_path)
{
    vector<String> filenames;
    glob(dataset_path + "*.mp4", filenames, true);

    namespace fs = std::filesystem;
    fs::path output_directory("output");
    fs::create_directories(output_directory);

    for (String filename : filenames)
    {
        clear_input_video_info();

        build_output_frames(filename, output_frames);

        fs::path output_path;
        output_path /= output_directory;
        output_path /= fs::path(filename).filename();

        cout << "Creating " << output_path.string() << "." << endl;
        build_video(output_frames, output_path.string());

        break;  // TODO remove
    }
}

void video_builder::build_output_frames(string filename, vector<Mat> &output_frames)
{
    VideoCapture input_video(filename);
    if (!input_video.isOpened())
        return;

    input_video_codec = static_cast<int>(input_video.get(CAP_PROP_FOURCC));
    input_video_fps = input_video.get(CAP_PROP_FPS);
    input_video_size = Size(static_cast<int>(input_video.get(CAP_PROP_FRAME_WIDTH)),
                  static_cast<int>(input_video.get(CAP_PROP_FRAME_HEIGHT)));

    Mat first_frame, frame;
    input_video.read(first_frame);
    playing_field_localizer pl_field_loc;
    pl_field_loc.localize(first_frame);

    balls_localizer balls_loc(pl_field_loc.get_localization());
    balls_loc.localize(first_frame);

    Ptr<legacy::MultiTracker> multiTracker = legacy::MultiTracker::create();

    // Initialize the trackers for each detected bounding box
    for (const auto &roi : balls_loc.get_bounding_boxes())
    {
        multiTracker->add(legacy::TrackerCSRT::create(), first_frame, roi);
    }

    // Load minimap
    Mat pool_table_map = imread("pool_table.png");
    Mat trajectories = pool_table_map.clone();
    minimap mini(pl_field_loc.get_localization(), balls_loc.get_localization());

    vector<Point> initial_balls_pos;
    vector<int> solids_indeces;
    vector<int> stripes_indeces;
    int black_index;
    int cue_index;
    mini.get_balls_pos(multiTracker->getObjects(), initial_balls_pos);
    mini.draw_initial_minimap(initial_balls_pos, balls_loc.get_localization(), solids_indeces, stripes_indeces, black_index, cue_index, first_frame, pool_table_map);
    imshow("initial minimap", pool_table_map);
    vector<Rect2d> old_balls_bounding_boxes = multiTracker->getObjects();

    while (input_video.read(frame))
    {

        multiTracker->update(frame);

        vector<Point> old_balls_pos;
        mini.get_balls_pos(old_balls_bounding_boxes, old_balls_pos);

        vector<Point> current_balls_pos;
        mini.get_balls_pos(multiTracker->getObjects(), current_balls_pos);

        mini.draw_minimap(old_balls_pos, current_balls_pos, solids_indeces, stripes_indeces, black_index, cue_index, frame, trajectories, pool_table_map);
        old_balls_bounding_boxes = multiTracker->getObjects();

        Mat output_frame;
        build_output_frame(frame, pool_table_map, output_frame);
        output_frames.push_back(output_frame);
    }
}

void video_builder::build_output_frame(const Mat &frame, const Mat &minimap, Mat &dst)
{
    dst = frame.clone();

    const float RESIZE_SCALE = 0.333;
    Mat resized_minimap;
    resize(minimap, resized_minimap, Size(), RESIZE_SCALE, RESIZE_SCALE);

    const int MINIMAP_DISPLAY_OFFSET = 20;
    int x_offset = MINIMAP_DISPLAY_OFFSET;
    int y_offset = dst.rows - resized_minimap.rows - MINIMAP_DISPLAY_OFFSET;

    // Ensure the minimap fits within the destination frame
    if (y_offset < 0 || resized_minimap.cols > dst.cols || resized_minimap.rows > dst.rows)
    {
        std::cerr << "Error: Minimap is too large to fit in the destination frame with the given offset." << std::endl;
        return;
    }

    resized_minimap.copyTo(dst(Rect(x_offset, y_offset, resized_minimap.cols, resized_minimap.rows)));
}

void video_builder::clear_input_video_info()
{
    output_frames.clear();
    input_video_fps = -1;
    input_video_size = Size(-1, -1);
}
