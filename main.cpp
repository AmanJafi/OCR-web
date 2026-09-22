#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

const int GRID = 28;
const int SCALE = 10;

Mat canvas(GRID * SCALE, GRID * SCALE, CV_8UC1, Scalar(0));
bool drawing = false;

void drawPixel(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) drawing = true;
    if (event == EVENT_LBUTTONUP) drawing = false;

    if (drawing) {
        int px = x / SCALE;
        int py = y / SCALE;

        if (px >= 0 && px < GRID && py >= 0 && py < GRID) {
            circle(canvas,
                   Point(px * SCALE + SCALE/2, py * SCALE + SCALE/2),
                   SCALE / 2, 
                   Scalar(255),
                   FILLED);
        }
    }
}

Mat preprocessForMNIST(const Mat& img) {
    Mat work;
    img.copyTo(work);

    threshold(work, work, 1, 255, THRESH_BINARY);
    

    vector<vector<Point>> contours;
    findContours(work, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return Mat::zeros(28, 28, CV_32F);
    }

    int max_area = 0, best_idx = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        int area = contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            best_idx = i;
        }
    }

    Rect bbox = boundingRect(contours[best_idx]);
    

    int w = bbox.width;
    int h = bbox.height;
    int size = max(w, h);
    
    Mat square = Mat::zeros(size, size, CV_8UC1);
    int offsetX = (size - w) / 2;
    int offsetY = (size - h) / 2;
    work(bbox).copyTo(square(Rect(offsetX, offsetY, w, h)));
    
    Mat resized;
    resize(square, resized, Size(20, 20));
    

    Mat finalImg;
    copyMakeBorder(resized, finalImg, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0));
    

    Moments m = moments(finalImg, true);
    if (m.m00 != 0) {
        double cx = m.m10 / m.m00;
        double cy = m.m01 / m.m00;
        double shiftX = 14.0 - cx;
        double shiftY = 14.0 - cy;
        Mat M = (Mat_<double>(2,3) << 1, 0, shiftX, 0, 1, shiftY);
        warpAffine(finalImg, finalImg, M, finalImg.size());
    }

    GaussianBlur(finalImg, finalImg, Size(3, 3), 0.5);
    finalImg.convertTo(finalImg, CV_32F, 1.0 / 255.0);
    return finalImg;
}

int predictDigit(const Mat& img) {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "digit");
    static Ort::SessionOptions session_options;
    static Ort::Session session(nullptr);
    static bool initialized = false;

    if (!initialized) {
        session_options.SetIntraOpNumThreads(1);
        session = Ort::Session(env, "handwriting_model.onnx", session_options);
        initialized = true;
    }

    Mat processed = preprocessForMNIST(img);


    vector<int64_t> input_shape = {1, 28, 28, 1};
    vector<float> input_tensor_values(28 * 28);

    int idx = 0;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            input_tensor_values[idx++] = processed.at<float>(i, j);
        }
    }

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {"input_1"};
    const char* output_names[] = {"output_0"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* output = output_tensors.front().GetTensorMutableData<float>();

    int predicted = 0;
    float max_prob = output[0];
    for (int i = 1; i < 10; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted = i;
        }
    }
    

    cout << "Confidence: " << fixed << setprecision(2) << max_prob * 100.0 << "%" << endl;
    
    return predicted;
}

int main() {
    namedWindow("Draw (P = Predict, C = Clear)", WINDOW_AUTOSIZE);
    setMouseCallback("Draw (P = Predict, C = Clear)", drawPixel);

    while (true) {
        imshow("Draw (P = Predict, C = Clear)", canvas);
        char key = (char)waitKey(1);

        if (key == 27) break;
        if (key == 'c' || key == 'C') canvas.setTo(0);

        if (key == 'p' || key == 'P') {
            cout << "Predicted digit: " << predictDigit(canvas) << endl;
        }
    }

    destroyAllWindows();
    return 0;
    }
