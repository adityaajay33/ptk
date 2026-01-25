#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <mutex>
#include "runtime/components/synthetic_camera.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"

int main(int argc, char** argv) {
    // Initialize ROS
    rclcpp::init(argc, argv);
    
    std::cout << "Starting SyntheticCamera -> TensorWriter pipeline test (Multi-threaded Scheduler)...\n";
    std::cout << "Main Thread ID: " << std::this_thread::get_id() << "\n\n";
    
    // Create runtime context for logging
    ptk::core::RuntimeContextOptions options;
    options.info_stream = stdout;
    options.error_stream = stderr;

    ptk::core::RuntimeContext context;
    context.Init(options);
    
    // Create Scheduler
    ptk::core::Scheduler scheduler;
    scheduler.Init(&context);

    // Create SyntheticCamera node
    rclcpp::NodeOptions cam_options;
    auto camera = std::make_shared<ptk::components::SyntheticCamera>(cam_options);
    
    // Setup data frame and output port
    ptk::data::Frame camera_frame;
    ptk::core::OutputPort<ptk::data::Frame> camera_output;
    camera_output.Bind(&camera_frame);
    
    // Connect camera to output port
    camera->BindOutput(&camera_output);
    
    // Add component to scheduler
    scheduler.AddComponent(camera.get());

    // Start Scheduler
    auto status = scheduler.Start();
    if (!status.ok()) {
        std::cerr << "Scheduler start failed: " << status.message() << "\n";
        return 1;
    }
    
    std::cout << "Scheduler started! Each component pinned to its own thread.\n";

    // Run scheduler in background thread
    std::thread scheduler_thread([&scheduler](){
        scheduler.RunLoop();
    });

    std::cout << "Collecting 10 frames...\n\n";
    
    // Open output file
    std::ofstream outfile("tensor_output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open tensor_output.txt\n";
        scheduler.Stop();
        scheduler_thread.join();
        rclcpp::shutdown();
        return 1;
    }
    
    // Collect frames loop
    int frames_collected = 0;
    while (frames_collected < 10) {
        
        // Since OutputPort::get() is non-blocking, we use a polling pattern.
        // We track the frame index to ensure we only process each new frame once.
        // The loop sleep ensures we don't consume 100% CPU while waiting.
        
        static int64_t last_frame_index = -1;
        const ptk::data::Frame* frame = camera_output.get();
        if (frame && frame->frame_index > last_frame_index && !frame->image.empty()) {
            
            // Lock the mutex for this frame instance before reading
            std::unique_lock<std::mutex> lock(scheduler.GetDataMutex((void*)frame));

            last_frame_index = frame->frame_index;
            frames_collected++;

             // Get tensor info
            const auto& shape = frame->image.shape();
            int H = shape.dim(0);
            int W = shape.dim(1);
            int C = shape.dim(2);
            
            const uint8_t* data = static_cast<const uint8_t*>(frame->image.buffer().data());

            // Write frame metadata to file
            outfile << "=== Frame " << frame->frame_index << " ===\n";
            outfile << "Dimensions: " << H << "x" << W << "x" << C << "\n";
            outfile << "Pixel Format: " << static_cast<int>(frame->pixel_format) << "\n";
            outfile << "Layout: " << static_cast<int>(frame->layout) << "\n";
            outfile << "Camera ID: " << frame->camera_id << " (synthetic)\n";
            outfile << "Timestamp: " << frame->timestamp_ns << " ns\n";
            
            // Write first 100 pixel values as sample
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
            std::cout << "[OBSERVER][THREAD " << std::this_thread::get_id() << "] Received Frame " << frame->frame_index 
                      << " (" << H << "x" << W << "x" << C << ") "
                      << "Top-left RGB: (" 
                      << static_cast<int>(data[0]) << "," 
                      << static_cast<int>(data[1]) << "," 
                      << static_cast<int>(data[2]) << ")"
                      << " (wrote to file)\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    outfile.close();
    std::cout << "\nDone! Tensor data written to tensor_output.txt\n";
    std::cout << "The synthetic camera generates a test pattern:\n";
    std::cout << "  - Red channel: horizontal gradient (0->255 left to right)\n";
    std::cout << "  - Green channel: vertical gradient (0->255 top to bottom)\n";
    std::cout << "  - Blue channel: changes with frame number\n";
    
    scheduler.Stop();
    if (scheduler_thread.joinable()) {
        scheduler_thread.join();
    }
    
    std::cout << "Exiting.\n";
    rclcpp::shutdown();
    
    return 0;
}
