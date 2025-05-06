#ifndef PROJECT_H
#define PROJECT_H

#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

struct image_channels_bgr {
    Mat B;
    Mat G;
    Mat R;
};

struct image_channels_hsv {
    Mat H;
    Mat S;
    Mat V;
};

struct neighborhood_structure {
    int size;
    int* di;
    int* dj;
};

struct contour_info {
    vector<Point> border;
    vector<int> dir_vector;
    int label;
};

struct grayscale_mapping {
    uchar* grayscale_values;
    int count_grayscale_values;
};

image_channels_bgr break_channels(Mat source);
bool IsInside(Mat img, int i, int j);
Mat dilation(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat erosion(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat opening(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat closing(Mat source, neighborhood_structure neighborhood, int no_iter);
Mat complementary_mat(Mat source);
Mat intersection_mat(Mat A, Mat B);
Mat union_mat(Mat A, Mat B);
Mat region_filling(Mat source, neighborhood_structure neighborhood, Point start);
Mat find_red_color(image_channels_bgr bgr_channels);
Mat find_blue_color(image_channels_bgr bgr_channels);
Mat find_white_color(image_channels_bgr bgr_channels);
image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels);
void display_hsv_channels(image_channels_hsv hsv_channels);
Mat detect_white_from_saturation(Mat saturation);
bool is_white(Mat& image, int i, int j);
Mat two_pass_labeling(Mat source, neighborhood_structure neighborhood);
contour_info extract_contour(Mat source, Point P_0, neighborhood_structure neighborhood);
Point find_first_point(Mat source, int label);
Mat find_shapes(Mat labels, Mat original_image, neighborhood_structure neighborhood, neighborhood_structure neighborhood2);
int compute_area(Mat source);
bool equal_mat(Mat A, Mat B);
vector<Point> approximate_polygon(const vector<Point>& points, double epsilon);
#endif