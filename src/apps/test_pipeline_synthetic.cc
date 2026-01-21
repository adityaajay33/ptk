#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include "runtime/components/synthetic_camera.h"
#include "runtime/core/runtime_context.h"

int main(int argc, char** argv) {
    // Initialize ROS
    rclcpp::init(argc, argv);
    
    std::cout << "Starting SyntheticCamera -> TensorWriter pipeline test...\n";
    
    // Create runtime context for logging
    ptk::core::RuntimeContext context;
    
    // Create SyntheticCamera node
    rclcpp::NodeOptions cam_options;
    auto camera = std::make_shared<ptk::components::SyntheticCamera>(cam_options);
    
    // Setup data frame and output port
    ptk::data::Frame camera_frame;
    ptk::core::OutputPort<ptk::data::Frame> camera_output;
    camera_output.Bind(&camera_frame);
    
    // Connect camera to output port
    camera->BindOutput(&camera_output);
    
    // Initialize and start camera
    auto status = camera->Init(&context);
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
    
    std::cout << "SyntheticCamera started successfully!\n";
    std::cout << "Generating 10 frames and writing tensor info to file...\n\n";
    
    // Open output file
    std::ofstream outfile("tensor_output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open tensor_output.txt\n";
        camera->Stop();
        rclcpp::shutdown();
        return 1;
    }
    
    // Generate and write 10 frames
    for (int i = 0; i < 10; i++) {
        // Trigger camera to generate synthetic frame
        camera->Tick();
        
        // Read frame from port (zero-copy!)
        const ptk::data::Frame* frame = camera_output.get();
        
        if (frame == nullptr || frame->image.empty()) {
            std::cerr << "Frame " << i << ": Failed to generate\n";
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
        outfile << "Camera ID: " << frame->camera_id << " (synthetic)\n";
        outfile << "Timestamp: " << frame->timestamp_ns << " ns\n";
        
        // Write first 100 pixel values as sample
        const uint8_t* data = static_cast<const uint8_t*>(frame->image.buffer().data());
        outfile << "First 100 pixel values: ";
        for (int j = 0; j < std::min(100, H * W * C); j++) {
            outfile << static_cast<int>(data[j]) << " ";
        }
        outfile << "\n";
        
        // Write corner pixels to show the test pattern
        outfile << "Top-left corner (R,G,B): " 
                << static_cast<int>(data[0]) << "," 
                << static_cast<int>(data[1]) << "," 
                << static_cast<int>(data[2]) << "\n";
        
        int bottom_right_idx = (H-1) * W * C + (W-1) * C;
        outfile << "Bottom-right corner (R,G,B): " 
                << static_cast<int>(data[bottom_right_idx]) << "," 
                << static_cast<int>(data[bottom_right_idx + 1]) << "," 
                << static_cast<int>(data[bottom_right_idx + 2]) << "\n";
        outfile << "\n";
        
        // Console output
        std::cout << "Frame " << frame->frame_index 
                  << ": " << H << "x" << W << "x" << C 
                  << " - Top-left RGB: (" 
                  << static_cast<int>(data[0]) << "," 
                  << static_cast<int>(data[1]) << "," 
                  << static_cast<int>(data[2]) << ")"
                  << " (wrote to file)\n";
        
        // Small delay between frames
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    outfile.close();
    std::cout << "\nDone! Tensor data written to tensor_output.txt\n";
    std::cout << "The synthetic camera generates a test pattern:\n";
    std::cout << "  - Red channel: horizontal gradient (0->255 left to right)\n";
    std::cout << "  - Green channel: vertical gradient (0->255 top to bottom)\n";
    std::cout << "  - Blue channel: changes with frame number\n";
    
    // Cleanup
    camera->Stop();
    rclcpp::shutdown();
    
    return 0;
}
