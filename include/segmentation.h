#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**
 * @brief Performs k-means clustering on an image for image segmentation.
 *
 * This function applies the k-means clustering algorithm to the input image (`src`) and stores the resulting segmented image in `dst`.
 * The number of clusters (centroids) is specified by the `centroids` parameter. The output image (`dst`) will have pixel values
 * replaced by their corresponding cluster centroid values, effectively segmenting the image based on color similarity.
 *
 * @param src The source image to be segmented.
 * @param dst The destination image where the segmented output is stored.
 * @param centroids The number of clusters (centroids) to use for k-means clustering.
 */
void kmeans(const cv::Mat &src, cv::Mat &dst, int centroids);

/**
 * @brief Performs region growing segmentation on an image.
 *
 * This function applies region growing segmentation to the input image (`src`) using the provided seed points (`seeds`).
 * The resulting segmented image is stored in `dst`. The growth of the region is controlled by the threshold parameters
 * (`threshold_0`, `threshold_1`, `threshold_2`) which define the maximum allowed difference in pixel values for the
 * region to grow. The output image (`dst`) will have regions marked based on the similarity to the seed points.
 *
 * @param src The source image to be segmented.
 * @param dst The destination image where the segmented output is stored.
 * @param seeds A vector of points to be used as seed points for region growing.
 * @param threshold_0 The threshold for the first channel (e.g., red in RGB) to control region growth.
 * @param threshold_1 The threshold for the second channel (e.g., green in RGB) to control region growth.
 * @param threshold_2 The threshold for the third channel (e.g., blue in RGB) to control region growth.
 */
void region_growing(const cv::Mat &src, cv::Mat &dst, const std::vector<cv::Point> &seeds, int threshold_0, int threshold_1, int threshold_2);

/**
 * @brief Performs region growing on a given source binary image starting from seed points and produces a binary mask.
 *
 * This function initializes a destination image with zeros, then iteratively grows regions from the given seed points.
 * A pixel is added to a region if the color is the same as the current pixel.
 *
 * @param src The source image to be segmented.
 * @param dst The destination image where the segmented output is stored.
 * @param seeds A vector of points to be used as seed points for region growing.
 */
void mask_region_growing(const cv::Mat &src, cv::Mat &dst, const std::vector<cv::Point> &seeds);

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
cv::Vec3b get_playing_field_color(const cv::Mat &src, float radius);

#endif