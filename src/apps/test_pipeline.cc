#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include "sensors/mac_camera.h"
#include "runtime/core/runtime_context.h"

int main(int argc, char** argv) {
    // Initialize ROS
    rclcpp::init(argc, argv);
    
    std::cout << "Starting MacCamera -> TensorWriter pipeline test...\n";
    
    // Create runtime context for logging
    ptk::core::RuntimeContext context;
    
    // Create MacCamera node
    rclcpp::NodeOptions cam_options;
    cam_options.append_parameter_override("device_index", 0);
    auto camera = std::make_shared<ptk::sensors::MacCamera>(cam_options);
    
    // Setup data frame and output port
    ptk::data::Frame camera_frame;
    ptk::core::OutputPort<ptk::data::Frame> camera_output;
    camera_output.Bind(&camera_frame);
    
    // Connect camera to output port
    camera->BindOutput(&camera_output);
    
    // Initialize and start camera
    auto status = camera->Init();
    if (!status.ok()) {
        std::cerr << "Camera Init failed: " << status.message() << "\n";
        rclcpp::shutdown();
        return 1;
    }
    
    status = camera->Start();
    if (!status.ok()) {
        std::cerr << "Camera Start failed: " << status.message() << "\n";
        rclcpp::shutdown();
        return 1;
    }
    
    std::cout << "Camera started successfully!\n";
    std::cout << "Capturing 10 frames and writing tensor info to file...\n\n";
    
    // Open output file
    std::ofstream outfile("tensor_output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open tensor_output.txt\n";
        camera->Stop();
        rclcpp::shutdown();
        return 1;
    }
    
    // Capture and write 10 frames
    for (int i = 0; i < 10; i++) {
        // Trigger camera to capture frame
        camera->Tick();
        
        // Read frame from port (zero-copy!)
        const ptk::data::Frame* frame = camera_output.get();
        
        if (frame == nullptr || frame->image.empty()) {
            std::cerr << "Frame " << i << ": Failed to capture\n";
            continue;
        }
        
        // Get tensor info
        const auto& shape = frame->image.shape();
        int H = shape.dim(0);
        int W = shape.dim(1);
        int C = shape.dim(2);
        
        // Write frame metadata to file
        outfile << "=== Frame " << frame->frame_index << " ===\n";
        outfile << "Dimensions: " << H << "x" << W << "x" << C << "\n";
        outfile << "Pixel Format: " << static_cast<int>(frame->pixel_format) << "\n";
        outfile << "Layout: " << static_cast<int>(frame->layout) << "\n";
        outfile << "Camera ID: " << frame->camera_id << "\n";
        outfile << "Timestamp: " << frame->timestamp_ns << " ns\n";
        
        // Write first 100 pixel values as sample
        const uint8_t* data = static_cast<const uint8_t*>(frame->image.buffer().data());
        outfile << "First 100 pixel values: ";
        for (int j = 0; j < std::min(100, H * W * C); j++) {
            outfile << static_cast<int>(data[j]) << " ";
        }
        outfile << "\n\n";
        
        // Console output
        std::cout << "Frame " << frame->frame_index 
                  << ": " << H << "x" << W << "x" << C 
                  << " (wrote to file)\n";
        
        // Small delay between captures
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    outfile.close();
    std::cout << "\nDone! Tensor data written to tensor_output.txt\n";
    
    // Cleanup
    camera->Stop();
    rclcpp::shutdown();
    
    return 0;
}
