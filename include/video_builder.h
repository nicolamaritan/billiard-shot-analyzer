#ifndef VIDEO_BUILDER_H
#define VIDEO_BUILDER_H

#include <string>
#include <opencv2/core.hpp>

class video_builder
{
public:
    void build_videos(std::string dataset_path);

private:
    void build_output_frames(std::string video_filename, std::vector<cv::Mat> &output_frames);
    void build_output_frame(const cv::Mat &frame, const cv::Mat &minimap, cv::Mat &dst);
    void build_video(std::vector<cv::Mat> output_frames, std::string output_filename);
    void clear_input_video_info();

    std::vector<cv::Mat> output_frames;
    double input_video_fps;
    cv::Size input_video_size;
    int input_video_codec;
};

#endif