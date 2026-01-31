#include "runtime/components/synthetic_camera.h"

#include "runtime/core/runtime_context.h"
#include "runtime/core/timing_helper.h"
#include "runtime/core/metrics.h"
#include "runtime/data/frame.h"
#include "runtime/core/scheduler.h"
#include <thread>
#include <sstream>
#include <mutex>
#include <iostream>

namespace ptk::components
{

    SyntheticCamera::SyntheticCamera(const rclcpp::NodeOptions &options)
        : ComponentInterface("synthetic_camera", options),
          context_(nullptr),
          output_(nullptr),
          frame_index_(0),
          total_frames_generated_(0),
          frames_dropped_(0)
    {
    }

    void SyntheticCamera::BindOutput(core::OutputPort<data::Frame> *port)
    {
    output_ = port;
    }

    core::Status SyntheticCamera::Init(core::RuntimeContext *context)
    {
        if (context == nullptr)
        {
            return core::Status(core::StatusCode::kInvalidArgument, "Context is null");
        }
        context_ = context;
        frame_index_ = 0;
        return core::Status::Ok();
    }

    core::Status SyntheticCamera::Start()
    {
        if (output_ == nullptr || !output_->is_bound())
        {
            return core::Status(core::StatusCode::kFailedPrecondition, "Output not bound");
        }
        return core::Status::Ok();
    }

    core::Status SyntheticCamera::Stop()
    {
        // Log statistics
        auto stats = output_->GetStats();
        RCLCPP_INFO(this->get_logger(), 
                    "Camera Statistics: Generated=%zu, Pushed=%zu, Dropped=%zu (queue policy=%zu)",
                    total_frames_generated_, stats.total_pushed, frames_dropped_, stats.total_dropped);
        return core::Status::Ok();
    }

    void SyntheticCamera::Tick()
    {
        core::ScopedTimer timer(get_name());

        if (output_ == nullptr || !output_->is_bound())
        {
            context_->LogError("Unbound output");
            return;
        }

        // Generate synthetic 640x480 RGB image dimensions
        const int H = 480;
        const int W = 640;
        const int C = 3;

        // Create a new frame with owned data (no shared state!)
        data::Frame frame = data::Frame::CreateOwned(H, W, C, 
                                                     core::PixelFormat::kRgb8,
                                                     core::TensorLayout::kHwc);
        
        frame.frame_index = frame_index_;
        frame.timestamp_ns = context_->NowNanoseconds();
        frame.camera_id = -1;  // Synthetic camera

        core::MetricsCollector::Instance().RecordFrameAge(get_name(), 0.0);

        // Get pointer to the owned pixel data
        uint8_t* pixel_data = frame.owned_data->data();
        
        // Generate test pattern (gradient + frame counter)
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int idx = (y * W + x) * C;
                pixel_data[idx + 0] = (x * 255) / W;  // R gradient horizontal
                pixel_data[idx + 1] = (y * 255) / H;  // G gradient vertical
                pixel_data[idx + 2] = (frame_index_ * 10) % 256;  // B changes with frame
            }
        }

        total_frames_generated_++;

        core::MetricsCollector::Instance().IncrementFramesProcessed(get_name());

        RCLCPP_DEBUG(this->get_logger(), 
                     "[CAMERA][THREAD %lu] Generating Frame %d", 
                     std::hash<std::thread::id>{}(std::this_thread::get_id()), 
                     frame_index_);

        if (output_->is_bound())
        {
            auto queue_stats = output_->GetStats();
            core::MetricsCollector::Instance().RecordQueueMetrics(
                std::string(get_name()) + "_output", queue_stats);
        }

        if (!output_->Push(std::move(frame)))
        {
            frames_dropped_++;
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                  "Frame %d dropped by queue policy (total dropped: %zu)",
                                  frame_index_, frames_dropped_);
        }

        frame_index_++;

        // Pacing: 30 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

} // namespace ptk::components

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::SyntheticCamera)