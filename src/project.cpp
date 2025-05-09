#include "project.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

image_channels_bgr break_channels(Mat source){
    Mat copy = source.clone();

    int rows = source.rows;
    int cols = source.cols;
    Mat B = Mat(rows, cols, CV_8UC1);
    Mat G = Mat(rows, cols, CV_8UC1);
    Mat R = Mat(rows, cols, CV_8UC1);

    image_channels_bgr bgr_channels;

    Vec3b pixel;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pixel = copy.at<Vec3b>(i, j);
            B.at<uchar>(i, j) = pixel[0];
            G.at<uchar>(i, j) = pixel[1];
            R.at<uchar>(i, j) = pixel[2];
        }
    }

    bgr_channels.B = B;
    bgr_channels.G = G;
    bgr_channels.R = R;

    return bgr_channels;
}

bool IsInside(Mat img, int i, int j) {
    return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

Mat dilation(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat aux = source.clone();
    int rows = source.rows;
    int cols = source.cols;
    int size = neighborhood.size;
    int *di = neighborhood.di;
    int *dj = neighborhood.dj;

    for (int k = 0; k < no_iter; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (source.at<uchar>(i, j) == 0) {
                    for (int n = 0; n < size; n++) {
                        int ni = i + di[n];
                        int nj = j + dj[n];
                        if (IsInside(source, ni, nj)) {
                            aux.at<uchar>(ni, nj) = 0;
                        }
                    }
                }
            }
        }
    }
    return aux;
}

Mat erosion(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat aux = source.clone();
    int rows = source.rows;
    int cols = source.cols;
    int size = neighborhood.size;
    int *di = neighborhood.di;
    int *dj = neighborhood.dj;

    for (int k = 0; k < no_iter; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (source.at<uchar>(i, j) == 0) {
                    for (int n = 0; n < size; n++) {
                        int ni = i + di[n];
                        int nj = j + dj[n];
                        if (IsInside(source, ni, nj)
                            && source.at<uchar>(ni, nj) == 255) {
                            aux.at<uchar>(i, j) = 255;
                        }
                    }
                }
            }
        }
    }

    return aux;
}

Mat opening(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat aux = source.clone();

    aux = erosion(aux, neighborhood, no_iter);
    aux = dilation(aux, neighborhood, no_iter);

    return aux;
}

Mat closing(Mat source, neighborhood_structure neighborhood, int no_iter) {
    Mat aux = source.clone();

    aux = dilation(aux, neighborhood, no_iter);
    aux = erosion(aux, neighborhood, no_iter);

    return aux;
}

bool equal_mat(Mat A, Mat B) {
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (A.at<uchar>(i, j) != B.at<uchar>(i, j)) {
                return false;
            }
        }
    }

    return true;
}

Mat complementary_mat(Mat source) {
    Mat result = source.clone();
    for (int i = 0; i < source.rows; i++) {
        for (int j = 0; j < source.cols; j++) {
            if (result.at<uchar>(i, j) == 0) {
                result.at<uchar>(i, j) = 255;
            } else {
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}

Mat intersection_mat(Mat A, Mat B) {
    Mat result = Mat::ones(A.rows, B.cols, A.type()) * 255;
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (A.at<uchar>(i, j) == 0 && B.at<uchar>(i, j) == 0) {
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}

Mat union_mat(Mat A, Mat B) {
    Mat result = Mat::ones(A.rows, B.cols, A.type()) * 255;
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (A.at<uchar>(i, j) == 0 || B.at<uchar>(i, j) == 0) {
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}

Mat region_filling(Mat source, neighborhood_structure neighborhood, Point start) {
    Mat comp = complementary_mat(source);

    Mat x_prev = Mat::ones(source.rows, source.cols, source.type()) * 255;
    Mat x_current = Mat::ones(source.rows, source.cols, source.type()) * 255;

    x_current.at<uchar>(start.y, start.x) = 0;

    do {
        x_prev = x_current;
        x_current = dilation(x_current, neighborhood, 1);
        x_current = intersection_mat(x_current, comp);
    } while (!equal_mat(x_prev, x_current));

    x_current = union_mat(x_current, source);

    return x_current;
}

Mat find_white_color(image_channels_bgr bgr_channels) {
    Mat b = bgr_channels.B;
    Mat g = bgr_channels.G;
    Mat r = bgr_channels.R;
    Mat result = Mat::ones(b.rows, b.cols, b.type()) * 255;

    for (int i = 0; i < b.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            if (b.at<uchar>(i, j) > 200 && 
                g.at<uchar>(i, j) > 200 && 
                r.at<uchar>(i, j) > 200) {
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}

Mat find_red_color(image_channels_bgr bgr_channels) {
    Mat b = bgr_channels.B;
    Mat g = bgr_channels.G;
    Mat r = bgr_channels.R;
    Mat result = Mat::ones(b.rows, b.cols, b.type()) * 255;

    for (int i = 0; i < b.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            if (0 < b.at<uchar>(i, j) && b.at<uchar>(i, j) < 132 &&
                0 < g.at<uchar>(i, j) && g.at<uchar>(i, j) < 78 &&
                105 < r.at<uchar>(i, j) && r.at<uchar>(i, j) < 255) {
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}

Mat find_blue_color(image_channels_bgr bgr_channels) {
    Mat b = bgr_channels.B;
    Mat g = bgr_channels.G;
    Mat r = bgr_channels.R;
    Mat result = Mat::ones(b.rows, b.cols, b.type()) * 255;

    for (int i = 0; i < b.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            if (130 < b.at<uchar>(i, j) && b.at<uchar>(i, j) < 255 &&
                0 < g.at<uchar>(i, j) && g.at<uchar>(i, j) < 120 &&
                0 < r.at<uchar>(i, j) && r.at<uchar>(i, j) < 100) {
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}

Point find_first_point(Mat source, uchar label) {
    int rows = source.rows;
    int cols = source.cols;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (source.at<uchar>(i, j) == label) {
                return Point(j, i);
            }
        }
    }
    return Point(-1, -1);
}

contour_info extract_contour(Mat source, Point P_0, neighborhood_structure neighborhood) {
    int dir = 7;
    Point P_current = P_0;
    vector<Point> border;
    vector<int> dir_vector;

    border.push_back(P_0);

    do {
        if (dir % 2 == 0) {
            dir = (dir + 7) % 8;
        } else {
            dir = (dir + 6) % 8;
        }

        int next_i = P_current.y + neighborhood.di[dir];
        int next_j = P_current.x + neighborhood.dj[dir];
        Point next(next_j, next_i);

        while (IsInside(source, next_i, next_j) && source.at<uchar>(next_i, next_j) != 0) {
            dir = (dir + 1) % 8;
            next_i = P_current.y + neighborhood.di[dir];
            next_j = P_current.x + neighborhood.dj[dir];
            next = Point(next_j, next_i);
        }

        if (IsInside(source, next_i, next_j) && source.at<uchar>(next_i, next_j) == 0) {
            P_current = next;
            border.push_back(P_current);
            dir_vector.push_back(dir);
        }

    } while (!(border[border.size() - 1] == border[1] && border[border.size() - 2] == border[0] && border.size() > 2));

    return {border, dir_vector, 0};
}

Mat two_pass_labeling(Mat source, neighborhood_structure neighborhood) {
    int rows = source.rows;
    int cols = source.cols;
    Mat labels = Mat::zeros(rows, cols, CV_32SC1);
    int no_newlabels = 0;
    vector<vector<int>> edges(1000);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (source.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
                vector<int> L;
                
                for (int n = 0; n < neighborhood.size; n++) {
                    int ni = i + neighborhood.di[n];
                    int nj = j + neighborhood.dj[n];
                    
                    if (IsInside(source, ni, nj) && labels.at<int>(ni, nj) > 0) {
                        L.push_back(labels.at<int>(ni, nj));
                    }
                }

                if (L.empty()) {
                    no_newlabels++;
                    labels.at<int>(i, j) = no_newlabels;
                } else {
                    int min_label = L[0];
                    for (int label : L) {
                        min_label = min(min_label, label);
                    }
                    labels.at<int>(i, j) = min_label;

                    for (int label : L) {
                        if (label != min_label) {
                            edges[min_label].push_back(label);
                            edges[label].push_back(min_label);
                        }
                    }
                }
            }
        }
    }

    int new_label = 0;
    vector<int> new_labels(no_newlabels + 1, 0);
    
    for (int i = 1; i <= no_newlabels; i++) {
        if (new_labels[i] == 0) {
            new_label++;
            queue<int> q;
            new_labels[i] = new_label;
            q.push(i);
            
            while (!q.empty()) {
                int current = q.front();
                q.pop();
                
                for (int neighbor : edges[current]) {
                    if (new_labels[neighbor] == 0) {
                        new_labels[neighbor] = new_label;
                        q.push(neighbor);
                    }
                }
            }
        }
    }

    Mat result = Mat::zeros(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int current_label = labels.at<int>(i, j);
            if (current_label > 0) {
                result.at<uchar>(i, j) = (uchar)((new_labels[current_label] * 255) / new_label);
            }
        }
    }

    return result;
}

int compute_area(Mat source) {
    int area = 0;
    int rows = source.rows;
    int cols = source.cols;
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(source.at<uchar>(i, j) == 0) {
                area++;
            }
        }
    }
    return area;
}

double point_to_line_distance(Point p, Point line_start, Point line_end) {
    double p_x = p.x;
    double p_y = p.y;
    double start_x = line_start.x;
    double start_y = line_start.y;
    double end_x = line_end.x;
    double end_y = line_end.y;

    double vec1_x = p_x - start_x;
    double vec1_y = p_y - start_y;
    double vec2_x = end_x - start_x;
    double vec2_y = end_y - start_y;

    double dot_prod = vec1_x * vec2_x + vec1_y * vec2_y;
    double sq_length = vec2_x * vec2_x + vec2_y * vec2_y;
    double param = -1;

    if (sq_length != 0) {
        param = dot_prod / sq_length;
    }

    double proj_x, proj_y;

    if (param < 0) {
        proj_x = start_x;
        proj_y = start_y;
    } else if (param > 1) {
        proj_x = end_x;
        proj_y = end_y;
    } else {
        proj_x = start_x + param * vec2_x;
        proj_y = start_y + param * vec2_y;
    }

    double dx = p_x - proj_x;
    double dy = p_y - proj_y;

    return sqrt(dx * dx + dy * dy);
}

vector<Point> approximate_polygon(vector<Point>& points, double epsilon) {
    if (points.size() <= 2) {
        return points;
    }

    double max_dist = 0;
    int max_index = 0;
    Point start = points.front();
    Point end = points.back();

    for (int i = 1; i < points.size() - 1; i++) {
        double dist = point_to_line_distance(points[i], start, end);
        if (dist > max_dist) {
            max_dist = dist;
            max_index = i;
        }
    }

    if (max_dist > epsilon) {
        vector first_half(points.begin(), points.begin() + max_index + 1);
        vector second_half(points.begin() + max_index, points.end());
        
        vector<Point> rec1 = approximate_polygon(first_half, epsilon);
        vector<Point> rec2 = approximate_polygon(second_half, epsilon);
        
        vector<Point> result;
        result.insert(result.end(), rec1.begin(), rec1.end() - 1);
        result.insert(result.end(), rec2.begin(), rec2.end());
        return result;
    }

    return {start, end};
}

Mat find_shapes(Mat labels, Mat original_image, neighborhood_structure neighborhood, neighborhood_structure neighborhood2) {
    Mat original = original_image.clone();
    Mat result = Mat::ones(labels.size(), CV_8UC1) * 255;
    int rows = labels.rows;
    int cols = labels.cols;

    const int MIN_THRESHOLD = 30;

    set<uchar> unique_labels;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(labels.at<uchar>(i, j) > 0) {
                unique_labels.insert(labels.at<uchar>(i, j));
            }
        }
    }

    for(uchar label : unique_labels) {
        Mat object_mask = Mat::zeros(labels.size(), CV_8UC1);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(labels.at<uchar>(i, j) == label) {
                    object_mask.at<uchar>(i, j) = 0;
                } else {
                    object_mask.at<uchar>(i, j) = 255;
                }
            }
        }

        int area = compute_area(object_mask);
        if (area < MIN_THRESHOLD) {
            continue;
        }
        
        Point P_0 = find_first_point(labels, label);

        if(P_0.x != -1) {
            contour_info contour = extract_contour(object_mask, P_0, neighborhood);
            
            double perimeter = contour.border.size();
            double epsilon = 0.04 * perimeter;
            
            vector<Point> approx_curve = approximate_polygon(contour.border, epsilon);
            
            int min_x = cols, min_y = rows, max_x = 0, max_y = 0;
            for(Point point : contour.border) {
                min_x = min(min_x, point.x);
                min_y = min(min_y, point.y);
                max_x = max(max_x, point.x);
                max_y = max(max_y, point.y);
            }
            Point center((min_x + max_x) / 2, (min_y + max_y) / 2);
            
            Mat filled = region_filling(object_mask, neighborhood2, center);
            int area = compute_area(filled);
            
            bool is_valid_shape = false;
            const Scalar PURPLE = Scalar(255, 0, 255);
            
            printf("Shape %d - Area: %d, Vertices: %d\n",
                   label, area, (int)approx_curve.size());
            
            if(approx_curve.size() == 3) {
                printf("  -> Identified as TRIANGLE\n");
                is_valid_shape = true;
            } else if (approx_curve.size() == 4) {
                printf("  -> Identified as RECTANGLE\n");
                is_valid_shape = true;
            }
            else if(approx_curve.size() >= 8) {
                double circularity = 4 * CV_PI * area / (perimeter * perimeter);
                if(circularity > 0.7) {
                    printf("  -> Identified as CIRCLE\n");
                    is_valid_shape = true;
                }
            }

            if(is_valid_shape) {
                line(original, Point(min_x, min_y), Point(max_x, min_y), PURPLE, 2);
                line(original, Point(max_x, min_y), Point(max_x, max_y), PURPLE, 2);
                line(original, Point(max_x, max_y), Point(min_x, max_y), PURPLE, 2);
                line(original, Point(min_x, max_y), Point(min_x, min_y), PURPLE, 2);
            }
        }
    }
    
    return original;
}

image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels){
    int rows = bgr_channels.B.rows;
    int cols = bgr_channels.B.cols;

    Mat H = Mat(rows, cols, CV_32FC1);
    Mat S = Mat(rows, cols, CV_32FC1);
    Mat V = Mat(rows, cols, CV_32FC1);

    float r, g, b;
    float M, m, C;
    image_channels_hsv hsv_channels;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            r = (float)bgr_channels.R.at<uchar>(i, j) / 255;
            g = (float)bgr_channels.G.at<uchar>(i, j) / 255;
            b = (float)bgr_channels.B.at<uchar>(i, j) / 255;

            M = max(r, max(g, b));
            m = min(r, min(g, b));
            C = M - m;

            V.at<float>(i, j) = M;
            if (V.at<float>(i, j) != 0.0) {
                S.at<float>(i, j) = C / V.at<float>(i, j);
            } else {
                S.at<float>(i, j) = 0.0;
            }

            if (C != 0.0) {
                if (M == r) {
                    H.at<float>(i, j) = 60 * (g - b) / C;
                }
                if (M == g) {
                    H.at<float>(i, j) = 120 + 60 * (b - r) / C;
                }
                if (M == b) {
                    H.at<float>(i, j) = 240 + 60 * (r - g) / C;
                }
            } else {
                H.at<float>(i, j) = 0.0;
            }
            if (H.at<float>(i, j) < 0.0) {
                H.at<float>(i, j) += 360;
            }
        }
    }

    hsv_channels.H = H;
    hsv_channels.S = S;
    hsv_channels.V = V;

    return hsv_channels;
}

void display_hsv_channels(image_channels_hsv hsv_channels){
    int rows = hsv_channels.H.rows;
    int cols = hsv_channels.H.cols;
    Mat H_norm, S_norm, V_norm;
    Mat H, S, V;

    H_norm = Mat(rows, cols, CV_8UC1);
    S_norm = Mat(rows, cols, CV_8UC1);
    V_norm = Mat(rows, cols, CV_8UC1);

    H = hsv_channels.H;
    S = hsv_channels.S;
    V = hsv_channels.V;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            H_norm.at<uchar>(i, j) = H.at<float>(i, j) * 255 / 360;
            S_norm.at<uchar>(i, j) = S.at<float>(i, j) * 255;
            V_norm.at<uchar>(i, j) = V.at<float>(i, j) * 255;
        }
    }

    imshow("H", H_norm);
    imshow("S", S_norm);
    imshow("V", V_norm);
}

Mat detect_white_from_saturation(Mat saturation) {
    Mat result = Mat::ones(saturation.rows, saturation.cols, CV_8UC1) * 255;

    const float SATURATION_THRESHOLD = 0.3;
    
    for (int i = 0; i < saturation.rows; i++) {
        for (int j = 0; j < saturation.cols; j++) {
            if (saturation.at<float>(i, j) < SATURATION_THRESHOLD) {
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    
    return result;
}

