// Author: Nicola Maritan 2121717
#ifndef VIDEO_BUILDER_H
#define VIDEO_BUILDER_H

#include <string>
#include <opencv2/core.hpp>

class video_builder
{
public:
    /**
     * @brief Processes a vector of input video files of games and produces a set of output video with the minimap,
     * saved as files.
     *
     * @param filename The name of the input video file.
     */
    void build_videos(const std::string &dataset_path);

    /**
     * @brief Processes an input video file of a game and produces an output video with the minimap, saved as file.
     *
     * @param filename The name of the input video file.
     */
    void build_video(const std::string &filename);

private:
    /**
     * @brief Builds output frames from an input video file.
     *
     * @param filename The name of the input video file.
     * @param output_frames A vector to store the output frames.
     */
    void build_output_frames(const std::string &video_filename, std::vector<cv::Mat> &output_frames);

    /**
     * @brief Combines a video frame and a minimap into a single output frame.
     *
     * @param frame The original video frame.
     * @param minimap The minimap to be included in the output frame.
     * @param dst The destination Mat object to store the combined output frame.
     */
    void build_output_frame(const cv::Mat &frame, const cv::Mat &minimap, cv::Mat &dst);

    /**
     * @brief Constructs a video file from a sequence of output frames.
     *
     * @param output_frames Vector of frames to be written to the video.
     * @param output_filename Name of the output video file.
     */
    void build_video_from_output_frames(const std::vector<cv::Mat> &output_frames, const std::string &output_filename);

    /**
     * @brief Clears the input video information.
     */
    void clear_input_video_info();

    /**
     * @brief Rescales a bounding box by a given scale factor and ensures it does not exceed a maximum size.
     *
     * @param bbox The original bounding box to be rescaled.
     * @param scale The scaling factor to be applied to the bounding box.
     * @param max_size The maximum allowed size for the bounding box.
     * @return A new Rect object representing the rescaled bounding box.
     */
    cv::Rect rescale_bounding_box(const cv::Rect &bbox, float scale, int max_size);

    // Vector to store output frames
    std::vector<cv::Mat> output_frames;

    // Frame rate of the input video
    double input_video_fps;

    // Size of the input video frames
    cv::Size input_video_size;

    // Codec used for the input video
    int input_video_codec;
};

#endif