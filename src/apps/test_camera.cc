#include "sensors/camera.h"
#include "runtime/core/queue_policy.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <rclcpp/rclcpp.hpp>

int main() {
    rclcpp::init(0, nullptr);

    std::cout << "\n========================================\n";
    std::cout << "Camera Producer-Consumer Test\n";
    std::cout << "========================================\n\n";

    // Create Camera (Producer)
    rclcpp::NodeOptions options;
    options.append_parameter_override("device_index", 0);
    ptk::sensors::Camera camera(options);

    // Initialize and start camera
    auto st = camera.Init();
    if (!st.ok()) {
        std::cout << "Camera Init failed: " << st.message() << "\n";
        return 1;
    }

    st = camera.Start();
    if (!st.ok()) {
        std::cout << "Camera Start failed: " << st.message() << "\n";
        return 1;
    }

    std::cout << "Camera started successfully!\n";

    // Create queue with Latest-Only policy (capacity=1)
    auto frame_queue = std::make_shared<ptk::core::BoundedQueue<ptk::data::Frame>>(
        1,
        ptk::core::QueuePolicy::kLatestOnly
    );

    // Create ports
    ptk::core::OutputPort<ptk::data::Frame> camera_output;
    camera_output.Bind(frame_queue);

    ptk::core::InputPort<ptk::data::Frame> consumer_input;
    consumer_input.Bind(frame_queue);

    // Bind camera output
    camera.BindOutput(&camera_output);

    std::cout << "Queue connected (Latest-Only policy, capacity=1)\n";
    std::cout << "Capturing and consuming 10 frames...\n\n";

    // Producer-Consumer loop
    for (int i = 0; i < 10; i++) {
        // PRODUCER: Camera captures and pushes frame
        camera.Tick();

        // Small delay to ensure frame is in queue
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // CONSUMER: Pop frame from queue
        auto frame_opt = consumer_input.Pop(std::chrono::milliseconds(100));
        
        if (!frame_opt) {
            std::cout << "Frame " << i << ": Timeout - no frame available\n\n";
            continue;
        }

        // We now own this frame (moved from queue)
        const ptk::data::Frame& frame = *frame_opt;

        if (frame.image.empty()) {
            std::cout << "Frame " << i << ": Empty frame\n\n";
            continue;
        }

        // Get frame info
        const auto& shape = frame.image.shape();
        int H = shape.dim(0);
        int W = shape.dim(1);
        int C = shape.dim(2);

        std::cout << "=== Frame " << frame.frame_index << " ===\n";
        std::cout << "Dimensions: " << H << "x" << W << "x" << C << "\n";
        std::cout << "Pixel Format: RGB8\n";
        std::cout << "Layout: HWC\n\n";

        // Display first 20 pixels as 2D array with RGB values
        const uint8_t* data = static_cast<const uint8_t*>(frame.image.buffer().data());
        int pixels_to_show = std::min(20, H * W);

        std::cout << "First " << pixels_to_show << " pixels [R, G, B]:\n";
        std::cout << "---------------------------------------------\n";

        for (int p = 0; p < pixels_to_show; p++) {
            int idx = p * C;
            int r = static_cast<int>(data[idx + 0]);
            int g = static_cast<int>(data[idx + 1]);
            int b = static_cast<int>(data[idx + 2]);

            // Calculate row and column position
            int row = p / W;
            int col = p % W;

            std::cout << "Pixel[" << row << "][" << col << "]: "
                      << "[" << r << ", " << g << ", " << b << "]\n";
        }
        std::cout << "\n";

        // Small delay between frames
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Print queue statistics
    auto stats = frame_queue->GetStats();
    std::cout << "========================================\n";
    std::cout << "Queue Statistics:\n";
    std::cout << "  Pushed:  " << stats.total_pushed << "\n";
    std::cout << "  Popped:  " << stats.total_popped << "\n";
    std::cout << "  Dropped: " << stats.total_dropped << "\n";
    std::cout << "  Current Depth: " << stats.current_depth << "\n";
    std::cout << "========================================\n\n";

    // Cleanup
    camera.Stop();
    rclcpp::shutdown();

    std::cout << "Test completed successfully!\n";
    return 0;
}
