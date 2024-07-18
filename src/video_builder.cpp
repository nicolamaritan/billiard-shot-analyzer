// Author: Nicola Maritan
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
namespace fs = std::filesystem;


void video_builder::build_videos(const string &dataset_path)
{
    vector<String> filenames;
    glob(dataset_path + "*.mp4", filenames, true);

    fs::path output_directory("output");
    fs::path videos_directory("videos");
    output_directory /= videos_directory;
    fs::create_directories(output_directory);

    for (String filename : filenames)
    {
        clear_input_video_info();

        build_output_frames(filename, output_frames);

        fs::path output_path;
        output_path /= output_directory;
        output_path /= fs::path(filename).filename();

        cout << "Creating " << output_path.string() << "." << endl;
        build_video_from_output_frames(output_frames, output_path.string());
    }
}



void video_builder::build_video(const string &filename)
{
    vector<String> filenames;

    fs::path output_directory("output");
    fs::path videos_directory("videos");
    output_directory /= videos_directory;
    fs::create_directories(output_directory);

    clear_input_video_info();
    build_output_frames(filename, output_frames);

    fs::path output_path;
    output_path /= output_directory;
    output_path /= fs::path(filename).filename();

    cout << "Creating " << output_path.string() << "." << endl;
    build_video_from_output_frames(output_frames, output_path.string());
}



void video_builder::build_output_frames(const string &filename, vector<Mat> &output_frames)
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

    Ptr<legacy::MultiTracker> multi_tracker = legacy::MultiTracker::create();

    // Initialize the trackers for each detected bounding box
    for (const auto &bbox : balls_loc.get_bounding_boxes())
    {
        /*
            The bounding boxes provided to the trackers are scaled by a factor
            greater than 1. This approach is adopted because we have observed that
            an enlarged bounding box enhances the tracker's ability to follow the
            ball accurately. Retaining the original detected bounding boxes often
            results in the tracker conflating adjacent or colliding balls. However,
            it is crucial that the scaling factor is not excessively large, as
            this would impair the tracker's ability to follow the ball effectively.
        */
        const float BOUNDING_BOX_RESCALE = 1.3;
        const int MAX_BOUNDING_BOX_SIZE = 30;
        multi_tracker->add(legacy::TrackerCSRT::create(), first_frame, rescale_bounding_box(bbox, BOUNDING_BOX_RESCALE, MAX_BOUNDING_BOX_SIZE));
    }

    // Load minimap
    const string MINIMAP_IMAGE_FILENAME = "pool_table.png";
    Mat pool_table_map = imread(MINIMAP_IMAGE_FILENAME);
    Mat trajectories = pool_table_map.clone();
    minimap mini(pl_field_loc.get_localization(), balls_loc.get_localization());

    vector<Point> initial_balls_pos;
    vector<int> solids_indeces;
    vector<int> stripes_indeces;
    int black_index;
    int cue_index;
    mini.get_balls_pos(multi_tracker->getObjects(), initial_balls_pos);
    mini.draw_initial_minimap(initial_balls_pos, balls_loc.get_localization(), solids_indeces, stripes_indeces, black_index, cue_index, first_frame, pool_table_map);
    vector<Rect2d> old_balls_bounding_boxes = multi_tracker->getObjects();

    while (input_video.read(frame))
    {

        multi_tracker->update(frame);

        vector<Point> old_balls_pos;
        mini.get_balls_pos(old_balls_bounding_boxes, old_balls_pos);

        vector<Point> current_balls_pos;
        mini.get_balls_pos(multi_tracker->getObjects(), current_balls_pos);

        mini.draw_minimap(old_balls_pos, current_balls_pos, solids_indeces, stripes_indeces, black_index, cue_index, frame, trajectories, pool_table_map);
        old_balls_bounding_boxes = multi_tracker->getObjects();

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



void video_builder::build_video_from_output_frames(const vector<Mat> &output_frames, const string &output_filename)
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



void video_builder::clear_input_video_info()
{
    output_frames.clear();
    input_video_fps = -1;
    input_video_size = Size(-1, -1);
}



Rect video_builder::rescale_bounding_box(const Rect &bbox, float scale, int max_size)
{
    int new_width = static_cast<int>(bbox.width * scale);
    int new_height = static_cast<int>(bbox.height * scale);

    if (new_width > max_size)
        new_width = max_size;
    if (new_height > max_size)
        new_height = max_size;

    // Calculate the new top-left corner to maintain the center of the bounding box
    int new_x = bbox.x + (bbox.width - new_width) / 2;
    int new_y = bbox.y + (bbox.height - new_height) / 2;

    return cv::Rect(new_x, new_y, new_width, new_height);
}