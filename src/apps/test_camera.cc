#include "sensors/mac_camera.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <rclcpp/rclcpp.hpp>

int main() {
    // Initialize ROS 2
    rclcpp::init(0, nullptr);
    
    // Create NodeOptions with device_index parameter
    rclcpp::NodeOptions options;
    options.append_parameter_override("device_index", 0);
    
    // Use FaceTime HD Camera
    ptk::sensors::MacCamera cam(options);

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

    // Warm up using raw OpenCV (bypass PTK)
    // Access underlying OpenCV capture directly
    cv::VideoCapture warm_cap(0, cv::CAP_AVFOUNDATION);

    if (warm_cap.isOpened()) {
        cv::Mat tmp;
        for (int i = 0; i < 60; i++) {
            warm_cap.read(tmp);
        }
        warm_cap.release();
    }

    // Now grab real frame through PTK
    ptk::data::Frame frame;
    st = cam.GetFrame(&frame);
    if (!st.ok()) {
        std::cout << "GetFrame failed: " << st.message() << "\n";
        return 1;
    }

    // Save frame as PNG
    int H = frame.image.shape().dim(0);
    int W = frame.image.shape().dim(1);
    int C = frame.image.shape().dim(2);

    cv::Mat img(H, W, (C == 3 ? CV_8UC3 : CV_8UC1),
                frame.image.buffer().data());

    cv::imwrite("captured_frame.png", img);

    std::cout << "Saved warmed frame to captured_frame.png\n";

    cam.Stop();
    
    // Shutdown ROS 2
    rclcpp::shutdown();
    
    return 0;
}