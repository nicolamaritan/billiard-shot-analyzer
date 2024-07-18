#include "performance_measurement.h"
#include "frame_segmentation.h"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

/**
 * @brief Calculate the Intersection over Union (IoU) for a specific class between two masks.
 *
 * @param predicted_mask The predicted mask.
 * @param ground_truth_mask The ground truth mask.
 * @param class_id The class ID for which to calculate the IoU.
 * @return The IoU value for the specified class.
 */
float get_class_iou(const cv::Mat &predicted_mask, const cv::Mat &ground_truth_mask, label_id class_id);



/**
 * @brief Get the match type and confidence for a predicted ball localization against the ground truth.
 *
 * @param predicted The predicted ball localization.
 * @param predicted_label The predicted label ID.
 * @param ground_truth The ground truth balls localization.
 * @return The match result.
 */
match get_match(const ball_localization &predicted, label_id predicted_label, const balls_localization ground_truth);



/**
 * @brief Calculate the Intersection over Union (IoU) between two rectangles.
 *
 * @param rect_1 The first rectangle.
 * @param rect_2 The second rectangle.
 * @return The IoU value.
 */
float get_iou(const Rect &rect_1, const Rect &rect_2);



/**
 * @brief Compute the average precision from a set of matches.
 *
 * @param matches A vector of match results.
 * @return The average precision value.
 */
float compute_average_precision(std::vector<match> &matches);



float get_class_iou(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask, label_id class_id)
{
    CV_Assert(found_mask.channels() == 1);
    CV_Assert(ground_truth_mask.channels() == 1);

    Mat found_class_mask, ground_truth_class_mask;
    inRange(found_mask, Scalar(class_id), Scalar(class_id), found_class_mask);
    inRange(ground_truth_mask, Scalar(class_id), Scalar(class_id), ground_truth_class_mask);
    Mat union_class_mask, intersection_class_mask;
    bitwise_or(found_class_mask, ground_truth_class_mask, union_class_mask);
    bitwise_and(found_class_mask, ground_truth_class_mask, intersection_class_mask);

    float union_area = static_cast<float>(countNonZero(union_class_mask));
    if (union_area == 0)
        return 1;
    float intersection_area = static_cast<float>(countNonZero(intersection_class_mask));
    return intersection_area / union_area;
}



float get_iou(const Rect &rect_1, const Rect &rect_2)
{
    Rect intersection_rect = rect_1 & rect_2;
    Rect union_rect = rect_1 | rect_2;
    return static_cast<float>(intersection_rect.area()) / static_cast<float>(union_rect.area());
}



match get_match(const ball_localization &predicted, label_id predicted_label, const balls_localization ground_truth)
{
    float current_iou = 0;

    float max_iou = get_iou(predicted.bounding_box, ground_truth.cue.bounding_box);
    label_id max_iou_id = label_id::cue;

    current_iou = get_iou(predicted.bounding_box, ground_truth.black.bounding_box);
    if (current_iou > max_iou)
    {
        max_iou = current_iou;
        max_iou_id = label_id::black;
    }

    for (ball_localization solid_localization : ground_truth.solids)
    {
        current_iou = get_iou(predicted.bounding_box, solid_localization.bounding_box);
        if (current_iou > max_iou)
        {
            max_iou = current_iou;
            max_iou_id = label_id::solids;
        }
    }

    for (ball_localization stripe_localization : ground_truth.stripes)
    {
        current_iou = get_iou(predicted.bounding_box, stripe_localization.bounding_box);
        if (current_iou > max_iou)
        {
            max_iou = current_iou;
            max_iou_id = label_id::stripes;
        }
    }

    const float IOU_THRESHOLD = 0.5;
    match returned_match;
    returned_match.type = max_iou > IOU_THRESHOLD && predicted_label == max_iou_id ? match_type::true_positive : match_type::false_positive;
    returned_match.confidence = predicted.confidence;

    return returned_match;
}



float compute_average_precision(std::vector<match> &matches)
{
    std::sort(matches.begin(), matches.end(), [](const match &a, const match &b)
              { return a.confidence > b.confidence; });

    // Calculate precision and recall
    int tp = 0, fp = 0;
    std::vector<float> precisions;
    std::vector<float> recalls;

    for (const auto &m : matches)
    {
        if (m.type == true_positive)
            tp++;
        else if (m.type == false_positive)
            fp++;
        float precision = static_cast<float>(tp) / (tp + fp);
        float recall = static_cast<float>(tp) / matches.size(); // Total positives assumed to be matches.size()
        precisions.push_back(precision);
        recalls.push_back(recall);
    }

    std::vector<float> recall_levels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<float> interpolated_precisions(11, 0.0);

    for (size_t i = 0; i < recall_levels.size(); i++)
    {
        float recall_level = recall_levels[i];
        float max_precision = 0.0;
        for (size_t j = 0; j < recalls.size(); j++)
        {
            if (recalls[j] >= recall_level)
                max_precision = std::max(max_precision, precisions[j]);
        }
        interpolated_precisions[i] = max_precision;
    }

    // Calculate average precision
    float average_precision = 0.0;
    for (float precision : interpolated_precisions)
    {
        average_precision += precision;
    }
    average_precision /= recall_levels.size();

    return average_precision;
}



float evaluate_balls_localization_dataset(const std::vector<balls_localization> &predicted_localizations, const std::vector<balls_localization> &ground_truth_localizations)
{
    CV_Assert(predicted_localizations.size() == ground_truth_localizations.size());

    vector<match> cue_matches;
    vector<match> black_matches;
    vector<match> solids_matches;
    vector<match> stripes_matches;

    for (int i = 0; i < predicted_localizations.size(); i++)
    {
        match cue_match = get_match(predicted_localizations.at(i).cue, label_id::cue, ground_truth_localizations.at(i));
        cue_matches.push_back(cue_match);

        match black_match = get_match(predicted_localizations.at(i).black, label_id::black, ground_truth_localizations.at(i));
        black_matches.push_back(black_match);

        for (ball_localization solid_loc : predicted_localizations.at(i).solids)
        {
            match solid_match = get_match(solid_loc, label_id::solids, ground_truth_localizations.at(i));
            solids_matches.push_back(solid_match);
        }

        for (ball_localization stripes_loc : predicted_localizations.at(i).stripes)
        {
            match stripe_match = get_match(stripes_loc, label_id::stripes, ground_truth_localizations.at(i));
            stripes_matches.push_back(stripe_match);
        }
    }

    float ap_cue = compute_average_precision(cue_matches);
    float ap_black = compute_average_precision(black_matches);
    float ap_solid = compute_average_precision(solids_matches);
    float ap_stripe = compute_average_precision(stripes_matches);

    int predicted_number_of_stripes = 0;
    int predicted_number_of_solids = 0;
    for (balls_localization loc : predicted_localizations)
    {
        predicted_number_of_solids += loc.solids.size();
        predicted_number_of_stripes += loc.stripes.size();
    }
    
    int ground_truth_number_of_solids = 0;
    int ground_truth_number_of_stripes = 0;
    for (balls_localization loc : predicted_localizations)
    {
        ground_truth_number_of_solids += loc.solids.size();
        ground_truth_number_of_stripes += loc.stripes.size();
    }

    // Handle cases in which it does not detect any item of a class and there are indeed no items of the class
    float solids_ap = 0;
    if (predicted_number_of_solids == 0 && ground_truth_number_of_solids == 0)
        solids_ap = 1;
    else
        solids_ap = compute_average_precision(solids_matches);

    float stripes_ap = 0;
    if (predicted_number_of_stripes == 0 && ground_truth_number_of_stripes == 0)
        stripes_ap = 1;
    else
        stripes_ap = compute_average_precision(stripes_matches);

    return (ap_cue + ap_black + ap_solid + ap_stripe) / 4;
}



float evaluate_balls_and_playing_field_segmentation_dataset(const std::vector<Mat> &predicted_masks, const std::vector<Mat> &ground_truth_masks)
{
    return 0;
}



float evaluate_balls_localization(const balls_localization &predicted, const balls_localization &ground_truth)
{
    match cue_match = get_match(predicted.cue, label_id::cue, ground_truth);
    // Compute precision and recall as follows since we are gauranteed to have one prediction and one ground truth for the cue
    float cue_tps = cue_match.type == match_type::true_positive ? 1 : 0;

    match black_match = get_match(predicted.black, label_id::black, ground_truth);
    // As above for cue ball
    float black_tps = cue_match.type == match_type::true_positive ? 1 : 0;

    float solids_tps = 0;
    vector<match> solids_matches;
    for (ball_localization localization : predicted.solids)
    {
        match solid_match = get_match(localization, label_id::solids, ground_truth);
        solids_matches.push_back(solid_match);
        solids_tps += cue_match.type == match_type::true_positive ? 1 : 0;
    }

    float stripes_tps = 0;
    vector<match> stripes_matches;
    for (ball_localization localization : predicted.stripes)
    {
        match stripe_match = get_match(localization, label_id::stripes, ground_truth);
        stripes_matches.push_back(stripe_match);
        stripes_tps += cue_match.type == match_type::true_positive ? 1 : 0;
    }

    /*cout << "cue tps: " << cue_tps << endl;
    cout << "black tps: " << black_tps << endl;
    cout << "solid tps: " << solids_tps << endl;
    cout << "stripe tps: " << stripes_tps << endl;
    cout << "--------------------------" << endl;
    */
    float cue_ap = cue_tps;
    float black_ap = black_tps;

    // Handle cases in which it does not detect any item of a class and there are indeed no items of the class
    float solids_ap = 0;
    if (predicted.solids.size() == 0 && ground_truth.solids.size() == 0)
        solids_ap = 1;
    else
        solids_ap = compute_average_precision(solids_matches);

    float stripes_ap = 0;
    if (predicted.stripes.size() == 0 && ground_truth.stripes.size() == 0)
        stripes_ap = 1;
    else
        stripes_ap = compute_average_precision(stripes_matches);

    float map = (cue_ap + black_ap + solids_ap + stripes_ap) / 4;

    return map;
}


float evaluate_balls_and_playing_field_segmentation(const cv::Mat &found_mask, const cv::Mat &ground_truth_mask)
{
    float iou_background = get_class_iou(found_mask, ground_truth_mask, label_id::background);
    float iou_cue = get_class_iou(found_mask, ground_truth_mask, label_id::cue);
    float iou_black = get_class_iou(found_mask, ground_truth_mask, label_id::black);
    float iou_solids = get_class_iou(found_mask, ground_truth_mask, label_id::solids);
    float iou_stripes = get_class_iou(found_mask, ground_truth_mask, label_id::stripes);
    float iou_playing_field = get_class_iou(found_mask, ground_truth_mask, label_id::playing_field);
    float mean_iou = (iou_background + iou_cue + iou_black + iou_solids + iou_stripes + iou_playing_field) / 6;

    /*cout << "background iou: " << iou_background << endl;
    cout << "cue iou: " << iou_cue << endl;
    cout << "black iou: " << iou_black << endl;
    cout << "solids iou: " << iou_solids << endl;
    cout << "stripes iou: " << iou_stripes << endl;
    cout << "playing iou: " << iou_playing_field << endl;
    cout << "mean iou: " << (iou_background + iou_cue + iou_black + iou_solids + iou_stripes + iou_playing_field) / 6 << endl;*/
    return mean_iou;
}



void get_balls_localization(const Mat &src, balls_localization &localization)
{
    playing_field_localizer plf_localizer;
    plf_localizer.localize(src);
    playing_field_localization plf_localization = plf_localizer.get_localization();

    balls_localizer blls_localizer(plf_localization);
    blls_localizer.localize(src);
    localization = blls_localizer.get_localization();
}



void load_ground_truth_localization(const string &filename, balls_localization &ground_truth_localization)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Could not open the file!" << endl;
        return;
    }

    int x, y, width, height, label_id;
    while (file >> x >> y >> width >> height >> label_id)
    {
        ball_localization ball;
        ball.bounding_box = Rect(x, y, width, height);

        switch (label_id)
        {
        case label_id::cue:
            ground_truth_localization.cue = ball;
            break;
        case label_id::black:
            ground_truth_localization.black = ball;
            break;
        case label_id::solids:
            ground_truth_localization.solids.push_back(ball);
            break;
        case label_id::stripes:
            ground_truth_localization.stripes.push_back(ball);
            break;
        default:
            cerr << "Unknown label_id: " << label_id << endl;
            break;
        }
    }

    file.close();
}