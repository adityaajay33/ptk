#include "sensors/camera.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <rclcpp/rclcpp.hpp>

int main() {
    rclcpp::init(0, nullptr);

    rclcpp::NodeOptions options;
    options.append_parameter_override("device_index", 0);

    ptk::sensors::Camera cam(options);

    auto st = cam.Init();
    if (!st.ok()) {
        std::cout << "Init failed: " << st.message() << "\n";
        return 1;
    }

    st = cam.Start();
    if (!st.ok()) {
        std::cout << "Start failed: " << st.message() << "\n";
        return 1;
    }

    ptk::data::Frame frame;
    st = cam.GetFrame(&frame);
    if (!st.ok()) {
        std::cout << "GetFrame failed: " << st.message() << "\n";
        return 1;
    }

    int H = frame.image.shape().dim(0);
    int W = frame.image.shape().dim(1);
    int C = frame.image.shape().dim(2);

    cv::Mat img(H, W, (C == 3 ? CV_8UC3 : CV_8UC1),
                frame.image.buffer().data());

    cv::imwrite("captured_frame.png", img);

    std::cout << "Saved frame to captured_frame.png (" << W << "x" << H << ")\n";

    cam.Stop();
    rclcpp::shutdown();

    return 0;
}
