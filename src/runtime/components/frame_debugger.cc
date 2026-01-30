// src/components/frame_debugger.cc
#include "runtime/components/frame_debugger.h"

#include <cstdint>
#include <string>
#include <mutex>

#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"
#include "runtime/data/frame.h"
#include "runtime/data/tensor.h" // whatever defines TensorView / TensorShape

namespace ptk::components
{

  FrameDebugger::FrameDebugger(const rclcpp::NodeOptions &options)
      : ComponentInterface("frame_debugger", options),
        context_(nullptr),
        input_(nullptr),
        tick_count_(0)
  {
  }

  void FrameDebugger::BindInput(core::InputPort<data::Frame> *port)
  {
    input_ = port;
  }

  core::Status FrameDebugger::Init(core::RuntimeContext *context)
  {
    if (context == nullptr)
    {
      return core::Status(core::StatusCode::kInvalidArgument, "Context is null");
    }
    context_ = context;
    tick_count_ = 0;
    return core::Status::Ok();
  }

  core::Status FrameDebugger::Start()
  {
    if (input_ == nullptr || !input_->is_bound())
    {
      return core::Status(core::StatusCode::kFailedPrecondition,
                          "Input not bound");
    }
    return core::Status::Ok();
  }

  core::Status FrameDebugger::Stop()
  {
    return core::Status::Ok();
  }

  void FrameDebugger::Tick()
  {
    ++tick_count_;

    if (input_ == nullptr || !input_->is_bound())
    {
      context_->LogError("Unbound input");
      return;
    }

    const data::Frame *frame = input_->get();
    if (frame == nullptr)
    {
      context_->LogError("Null frame");
      return;
    }

    // Lock the mutex for the input frame
    std::unique_lock<std::mutex> lock(scheduler_->GetDataMutex((void*)frame));

    const data::TensorView &image = frame->image;
    const data::TensorShape &shape = image.shape();

    if (shape.rank() != 3)
    {
      context_->LogError("Invalid image shape");
      return;
    }
  }

} // namespace ptk::components

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::FrameDebugger)
