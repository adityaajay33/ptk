// src/components/frame_debugger.cc
#include "runtime/components/frame_debugger.h"

#include <cstdint>
#include <string>

#include "runtime/core/runtime_context.h"
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
                          "FrameDebugger input not bound");
    }
    context_->LogInfo("FrameDebugger started.");
    return core::Status::Ok();
  }

  core::Status FrameDebugger::Stop()
  {
    if (context_)
    {
      std::string msg = "FrameDebugger stopped after " +
                        std::to_string(tick_count_) + " ticks.";
      context_->LogInfo(msg.c_str());
    }
    return core::Status::Ok();
  }

  void FrameDebugger::Tick()
  {
    ++tick_count_;

    if (input_ == nullptr || !input_->is_bound())
    {
      context_->LogError("FrameDebugger Tick with unbound input.");
      return;
    }

    const data::Frame *frame = input_->get();
    if (frame == nullptr)
    {
      context_->LogError("FrameDebugger Tick with null frame.");
      return;
    }

    // Direct member access, since Frame is a struct.
    const data::TensorView &image = frame->image;
    const data::TensorShape &shape = image.shape();

    if (shape.rank() != 3)
    {
      context_->LogError("FrameDebugger expected HxWxC image.");
      return;
    }

    int64_t height = shape.dim(0);
    int64_t width = shape.dim(1);
    int64_t channels = shape.dim(2);

    std::string msg = "FrameDebugger tick " +
                      std::to_string(tick_count_) +
                      ", size = " +
                      std::to_string(width) + "x" +
                      std::to_string(height) +
                      ", channels = " +
                      std::to_string(channels);
    context_->LogInfo(msg.c_str());
  }

} // namespace ptk::components

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::FrameDebugger)
