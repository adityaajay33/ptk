#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>
#include "runtime/components/synthetic_camera.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"
#include "runtime/core/queue_policy.h"

int main(int argc, char** argv) {
    // Initialize ROS
    rclcpp::init(argc, argv);
    
    std::cout << "Queue-Based Pipeline Test with Latest-Only Policy\n";
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
    
    auto frame_queue = std::make_shared<ptk::core::BoundedQueue<ptk::data::Frame>>(
        1,  // Capacity: single slot
        ptk::core::QueuePolicy::kLatestOnly  // Always overwrite with newest
    );
    
    std::cout << "Queue Configuration:\n";
    std::cout << "  Policy: Latest-Only (single slot, always fresh)\n";
    std::cout << "  Capacity: 1 frame\n";
    std::cout << "  Behavior: Old frames overwritten by new ones\n\n";
    
    // Setup queue-based output port
    ptk::core::OutputPort<ptk::data::Frame> camera_output;
    camera_output.Bind(frame_queue);
    
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
    
    std::cout << "Scheduler started! Camera running at 30 FPS in separate thread.\n";

    // Run scheduler in background thread
    std::thread scheduler_thread([&scheduler](){
        scheduler.RunLoop();
    });

    std::cout << "Collecting 10 frames from queue...\n\n";
    
    // Open output file
    std::ofstream outfile("tensor_output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open tensor_output.txt\n";
        scheduler.Stop();
        scheduler_thread.join();
        rclcpp::shutdown();
        return 1;
    }
    
    // Create an InputPort to consume from the same queue
    ptk::core::InputPort<ptk::data::Frame> observer_input;
    observer_input.Bind(frame_queue);
    
    // Collect frames loop
    int frames_collected = 0;
    int64_t last_frame_index = -1;
    
    while (frames_collected < 10) {
        // Try to pop a frame from the queue (blocking with timeout)
        auto frame_opt = observer_input.Pop(std::chrono::milliseconds(100));
        
        if (!frame_opt) {
            // Timeout - no frame available, retry
            std::cout << "[OBSERVER] Waiting for frame...\n";
            continue;
        }
        
        // We own this frame now - no mutex needed!
        const ptk::data::Frame& frame = *frame_opt;
        
        // Skip if we somehow got an old frame (shouldn't happen with Latest-Only)
        if (frame.frame_index <= last_frame_index) {
            continue;
        }
        
        last_frame_index = frame.frame_index;
        frames_collected++;
        
        // Calculate frame age
        int64_t now = context.NowNanoseconds();
        double frame_age_ms = (now - frame.timestamp_ns) / 1e6;
        
        // Get tensor info
        const auto& shape = frame.image.shape();
        int H = shape.dim(0);
        int W = shape.dim(1);
        int C = shape.dim(2);
        
        const uint8_t* data = static_cast<const uint8_t*>(frame.image.buffer().data());

        outfile << "=== Frame " << frame.frame_index << " ===\n";
        outfile << "Dimensions: " << H << "x" << W << "x" << C << "\n";
        outfile << "Pixel Format: " << static_cast<int>(frame.pixel_format) << "\n";
        outfile << "Layout: " << static_cast<int>(frame.layout) << "\n";
        outfile << "Camera ID: " << frame.camera_id << " (synthetic)\n";
        outfile << "Timestamp: " << frame.timestamp_ns << " ns\n";
        outfile << "Frame Age: " << frame_age_ms << " ms\n";
        
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
        std::cout << "[OBSERVER][THREAD " << std::this_thread::get_id() << "] "
                  << "Received Frame " << frame.frame_index 
                  << " (" << H << "x" << W << "x" << C << ") "
                  << "Age: " << frame_age_ms << "ms, "
                  << "Top-left RGB: (" 
                  << static_cast<int>(data[0]) << "," 
                  << static_cast<int>(data[1]) << "," 
                  << static_cast<int>(data[2]) << ")\n";
        
        // Simulate slow processing to test queue behavior
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    outfile.close();
    
    // Get final queue statistics
    auto stats = frame_queue->GetStats();
    
    std::cout << "\n=================================================\n";
    std::cout << "Test Complete!\n";
    std::cout << "=================================================\n";
    std::cout << "Queue Statistics:\n";
    std::cout << "  Total Pushed: " << stats.total_pushed << "\n";
    std::cout << "  Total Popped: " << stats.total_popped << "\n";
    std::cout << "  Total Dropped: " << stats.total_dropped << " (by Latest-Only policy)\n";
    std::cout << "  Current Depth: " << stats.current_depth << "\n\n";
    
    std::cout << "Tensor data written to tensor_output.txt\n";
    std::cout << "The synthetic camera generates a test pattern:\n";
    std::cout << "  - Red channel: horizontal gradient (0->255 left to right)\n";
    std::cout << "  - Green channel: vertical gradient (0->255 top to bottom)\n";
    std::cout << "  - Blue channel: changes with frame number\n\n";
    
    std::cout << "Key Benefits Demonstrated:\n";
    std::cout << "  ✓ No mutex contention (each frame is owned)\n";
    std::cout << "  ✓ Explicit drop policy (Latest-Only)\n";
    std::cout << "  ✓ Frame age tracking (freshness guarantee)\n";
    std::cout << "  ✓ Observable queue statistics\n\n";
    
    scheduler.Stop();
    if (scheduler_thread.joinable()) {
        scheduler_thread.join();
    }
    
    std::cout << "Exiting.\n";
    rclcpp::shutdown();
    
    return 0;
}
