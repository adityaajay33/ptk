#include "runtime/components/synthetic_camera.h"

#include "runtime/core/runtime_context.h"
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
          current_buffer_index_(0)
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
        return core::Status::Ok();
    }

    void SyntheticCamera::Tick()
    {
        if (output_ == nullptr || !output_->is_bound())
        {
            context_->LogError("Unbound output");
            return;
        }

        data::Frame *frame = output_->get();
        if (frame == nullptr)
        {
            context_->LogError("Null frame");
            return;
        }

        // swap to alternate buffer before writing
        current_buffer_index_ = 1 - current_buffer_index_;
        std::vector<uint8_t>& active_buffer = frame_buffer_[current_buffer_index_];

        // Lock the mutex for this frame instance
        std::unique_lock<std::mutex> lock(scheduler_->GetDataMutex(frame));

        std::cout << "[CAMERA][THREAD " << std::this_thread::get_id() << "] Generating Frame " << frame_index_ << "\n";

        // Generate synthetic 640x480 RGB image
        const int H = 480;
        const int W = 640;
        const int C = 3;
        const size_t num_bytes = H * W * C;
        
        // resize buffer - already alternated
        active_buffer.resize(num_bytes);
        
        // Generate test pattern (gradient + frame counter)
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int idx = (y * W + x) * C;
                active_buffer[idx + 0] = (x * 255) / W;  // R gradient horizontal
                active_buffer[idx + 1] = (y * 255) / H;  // G gradient vertical
                active_buffer[idx + 2] = (frame_index_ * 10) % 256;  // B changes with frame
            }
        }
        
        frame->image = data::TensorView(
            data::BufferView(active_buffer.data(), num_bytes, core::DeviceType::kCpu),
            core::DataType::kUint8,
            data::TensorShape({H, W, C})
        );
        
        frame->pixel_format = core::PixelFormat::kRgb8;
        frame->layout = core::TensorLayout::kHwc;
        frame->frame_index = frame_index_++;
        frame->timestamp_ns = context_->NowNanoseconds();
        frame->camera_id = -1;  // Synthetic camera

        // Pacing: 30 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

} // namespace ptk::components

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::SyntheticCamera)