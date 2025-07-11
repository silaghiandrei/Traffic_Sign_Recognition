#include <opencv2/opencv.hpp>
#include "src/project.h"
using namespace std;
using namespace cv;

int main(){
    int di_8[] = {0, -1, -1, -1, 0, 1, 1, 1};
    int dj_8[] = {1, 1, 0, -1, -1, -1, 0, 1};

    int di_4[] = { 0,-1, 0, 1};
    int dj_4[] = { 1, 0,-1, 0};

    neighborhood_structure n8 = {
        8,
        di_8,
        dj_8
    };

    neighborhood_structure n4 = {
        4,
        di_4,
        dj_4
    };

    Mat test_image = imread("../images/traffic_signs_13.jpg", IMREAD_COLOR);
    imshow("Original Image", test_image);

    image_channels_bgr bgr_channels = break_channels(test_image);
    image_channels_hsv hsv_channels = bgr_2_hsv(bgr_channels);

    Mat red_found = find_red_color(bgr_channels);
    red_found = closing(red_found, n8, 5);
    red_found = opening(red_found, n8, 3);
    //imshow("Red found", red_found);

    Mat blue_found = find_blue_color(bgr_channels);
    blue_found = closing(blue_found, n8, 5);
    blue_found = opening(blue_found, n8, 3);
    //imshow("Blue found", blue_found);

    Mat white_found = find_white_color(hsv_channels);
    white_found = opening(white_found, n8, 3);
    white_found = closing(white_found, n8, 5);
    //imshow("White found", white_found);

    Mat colors_found = union_mat(red_found, blue_found);
    colors_found = union_mat(colors_found, white_found);
    //imshow("Colors found", colors_found);

    Mat labeled_objects = two_pass_labeling(colors_found, n8);
    //imshow("Labeled objects", labeled_objects);

    Mat shapes = find_shapes(labeled_objects, test_image, n8, n4);
    imshow("Shapes", shapes);

    waitKey(0);
    return 0;
}