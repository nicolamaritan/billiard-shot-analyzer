#ifndef BALLS_LOCALIZER
#define BALLS_LOCALIZER

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class balls_localizer
{
public:
    /**
     * Localize the pool table.
     *
     * @param src The input image.
     */
    void localize(const cv::Mat &src, const cv::Mat &mask);

private:
    void localize_red_balls(const cv::Mat &src, cv::Mat &dst);
    void region_growing(const cv::Mat &src, cv::Mat &dst, const std::vector<cv::Point> &seeds, int threshold_0, int threshold_1, int threshold_2);
    void circles_masks(const std::vector<cv::Vec3f> &circles, std::vector<cv::Mat> &masks, cv::Size mask_size);
    void filter_empty_circles(const std::vector<cv::Vec3f> &circles, const std::vector<cv::Mat> &masks, const cv::Mat &segmentation_mask, std::vector<cv::Vec3f> &filtered_circles, std::vector<cv::Mat> &filtered_masks, float intersection_threshold);
    void filter_out_of_bound_circles(const std::vector<cv::Vec3f> &circles, const std::vector<cv::Mat> &masks, const cv::Mat &segmentation_mask, std::vector<cv::Vec3f> &filtered_circles, std::vector<cv::Mat> &filtered_masks, float intersection_threshold);
    /**
     * @brief Perform segmentation of the image based on color. One of the clusters should
     * contain the whole table, surrounded by different clusters.
     *
     * @param src The input image to segment.
     * @param dst The segmented image.
     */
    void segmentation(const cv::Mat &src, cv::Mat &dst);

    /**
     * @brief Return the estimated board color.
     *
     * It computes the color of the board by considering a circle of a given radius around
     * the center of the image and picking the median value.
     *
     * @param src Input image containing the board.
     * @param radius Radius from the image center in which to compute the board color.
     * @return the computed color of the board.
     */
    cv::Vec3b get_board_color(const cv::Mat &src, float radius);

    /**
     * @brief Performs non-maxima suppression on the connected components of the input image,
     *        keeping only the largest component.
     *
     * @param src Input binary image.
     * @param dst Output image where only the largest connected component is retained.
     */
    void non_maxima_connected_component_suppression(const cv::Mat &src, cv::Mat &dst);
};

#endif