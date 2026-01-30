#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include "runtime/components/synthetic_camera.h"
#include "runtime/components/inference_node.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"
#include "runtime/core/queue_policy.h"
#include "runtime/data/frame.h"
#include "tasks/task_output.h"

using namespace ptk;

/**
 * Simple component that consumes TaskOutput from a queue and logs results
 */
class DetectionLogger : public components::ComponentInterface
{
public:
    explicit DetectionLogger(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : ComponentInterface("detection_logger", options),
          context_(nullptr),
          input_(nullptr),
          frames_processed_(0)
    {
    }

    void BindInput(core::InputPort<tasks::TaskOutput> *in)
    {
        input_ = in;
    }

    core::Status Init(core::RuntimeContext *context) override
    {
        context_ = context;
        RCLCPP_INFO(this->get_logger(), "DetectionLogger initialized");
        return core::Status::Ok();
    }

    core::Status Start() override
    {
        RCLCPP_INFO(this->get_logger(), "DetectionLogger started");
        return core::Status::Ok();
    }

    core::Status Stop() override
    {
        RCLCPP_INFO(this->get_logger(), 
                    "DetectionLogger stopped. Processed %zu frames", frames_processed_);
        
        // Log queue statistics
        if (input_ && input_->is_bound())
        {
            auto stats = input_->GetStats();
            RCLCPP_INFO(this->get_logger(), 
                        "Output Queue Stats: Pushed=%zu, Popped=%zu, Dropped=%zu",
                        stats.total_pushed, stats.total_popped, stats.total_dropped);
        }
        
        return core::Status::Ok();
    }

    void Tick() override
    {
        if (!input_ || !input_->is_bound())
        {
            return;
        }

        // Block until result is available (uses condition variable - efficient!)
        auto result_opt = input_->Pop(std::chrono::milliseconds(100));
        if (!result_opt)
        {
            // Timeout - no result available
            return;
        }

        // We own this result - no mutex needed!
        const tasks::TaskOutput &result = *result_opt;

        if (!result.success)
        {
            RCLCPP_WARN(this->get_logger(), "Received failed inference result");
            return;
        }

        frames_processed_++;

        // Calculate result age
        int64_t now = context_->NowNanoseconds();
        double result_age_ms = (now - result.timestamp_ns) / 1e6;

        RCLCPP_INFO(this->get_logger(),
                    "Frame %ld: %zu detections, inference: %.1f ms, age: %.1f ms",
                    result.frame_index,
                    result.detections.size(),
                    result.inference_time_ms,
                    result_age_ms);

        // Log detections
        for (const auto &det : result.detections)
        {
            RCLCPP_INFO(this->get_logger(),
                        "  %s (%.2f) at [%.1f, %.1f, %.1f, %.1f]",
                        det.class_name.c_str(), det.confidence,
                        det.box.x1, det.box.y1, det.box.x2, det.box.y2);
        }
    }

private:
    core::RuntimeContext *context_;
    core::InputPort<tasks::TaskOutput> *input_;
    size_t frames_processed_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "  PTK Queue-Based Inference Pipeline Demo\n";
    std::cout << "========================================================\n";
    std::cout << "Pipeline: Camera -> Inference -> Logger\n";
    std::cout << "Queue Policies: Latest-Only (both stages)\n";
    std::cout << "========================================================\n\n";

    // Create runtime context
    core::RuntimeContextOptions options;
    options.info_stream = stdout;
    options.error_stream = stderr;

    core::RuntimeContext context;
    context.Init(options);

    // Create scheduler
    core::Scheduler scheduler;
    scheduler.Init(&context);

    // Create components
    auto camera = std::make_shared<components::SyntheticCamera>(rclcpp::NodeOptions());
    auto inference = std::make_shared<components::InferenceNode>(rclcpp::NodeOptions());
    auto logger = std::make_shared<DetectionLogger>(rclcpp::NodeOptions());

    // Create queues with Latest-Only policy
    // This ensures real-time behavior - always process the freshest data
    auto frame_queue = std::make_shared<core::BoundedQueue<data::Frame>>(
        1,  // Single slot
        core::QueuePolicy::kLatestOnly  // Overwrite old frames
    );

    auto result_queue = std::make_shared<core::BoundedQueue<tasks::TaskOutput>>(
        1,  // Single slot
        core::QueuePolicy::kLatestOnly  // Overwrite old results
    );

    std::cout << "Queue Configuration:\n";
    std::cout << "  Camera -> Inference: Latest-Only (capacity=1)\n";
    std::cout << "  Inference -> Logger: Latest-Only (capacity=1)\n\n";

    // Create ports
    core::OutputPort<data::Frame> camera_output;
    camera_output.Bind(frame_queue);

    core::InputPort<data::Frame> inference_input;
    inference_input.Bind(frame_queue);

    core::OutputPort<tasks::TaskOutput> inference_output;
    inference_output.Bind(result_queue);

    core::InputPort<tasks::TaskOutput> logger_input;
    logger_input.Bind(result_queue);

    // Bind ports to components
    camera->BindOutput(&camera_output);
    inference->BindInput(&inference_input);
    inference->BindOutput(&inference_output);
    logger->BindInput(&logger_input);

    // Add components to scheduler
    scheduler.AddComponent(camera.get());
    scheduler.AddComponent(inference.get());
    scheduler.AddComponent(logger.get());

    std::cout << "Starting scheduler with 3 threads (one per component)...\n\n";

    // Start scheduler (spawns threads)
    auto status = scheduler.Start();
    if (!status.ok())
    {
        std::cerr << "Failed to start scheduler: " << status.message() << "\n";
        return 1;
    }

    // Let the pipeline run for 5 seconds
    std::cout << "Pipeline running...\n";
    std::cout << "Press Ctrl+C to stop or wait 5 seconds\n\n";
    
    std::this_thread::sleep_for(std::chrono::seconds(5));

    std::cout << "\nStopping pipeline...\n\n";

    // Stop scheduler (joins threads and calls Stop() on all components)
    scheduler.Stop();

    // Print final statistics
    std::cout << "\n========================================================\n";
    std::cout << "  Pipeline Statistics\n";
    std::cout << "========================================================\n";
    
    auto frame_stats = frame_queue->GetStats();
    std::cout << "Frame Queue (Camera -> Inference):\n";
    std::cout << "  Pushed:  " << frame_stats.total_pushed << "\n";
    std::cout << "  Popped:  " << frame_stats.total_popped << "\n";
    std::cout << "  Dropped: " << frame_stats.total_dropped << " (by Latest-Only policy)\n";
    std::cout << "  Depth:   " << frame_stats.current_depth << "\n\n";

    auto result_stats = result_queue->GetStats();
    std::cout << "Result Queue (Inference -> Logger):\n";
    std::cout << "  Pushed:  " << result_stats.total_pushed << "\n";
    std::cout << "  Popped:  " << result_stats.total_popped << "\n";
    std::cout << "  Dropped: " << result_stats.total_dropped << " (by Latest-Only policy)\n";
    std::cout << "  Depth:   " << result_stats.current_depth << "\n\n";

    std::cout << "Key Benefits Demonstrated:\n";
    std::cout << "  ✓ No mutex contention (each thread owns its data)\n";
    std::cout << "  ✓ Explicit frame dropping (Latest-Only policy)\n";
    std::cout << "  ✓ Graceful degradation under load\n";
    std::cout << "  ✓ Observable queue statistics\n";
    std::cout << "  ✓ Frame age tracking\n";
    std::cout << "  ✓ Each component runs at its own pace\n";
    std::cout << "========================================================\n\n";

    rclcpp::shutdown();
    return 0;
}
