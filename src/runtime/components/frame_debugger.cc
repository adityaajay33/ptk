// src/components/frame_debugger.cc
#include "runtime/components/frame_debugger.h"

#include <cstdint>
#include <string>
#include <mutex>

#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"
#include "runtime/core/timing_helper.h"
#include "runtime/core/metrics.h"
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
    core::ScopedTimer timer(get_name());

    ++tick_count_;

    if (input_ == nullptr || !input_->is_bound())
    {
      context_->LogError("Unbound input");
      return;
    }

    if (input_->is_bound())
    {
      auto queue_stats = input_->GetStats();
      core::MetricsCollector::Instance().RecordQueueMetrics(
          std::string(get_name()) + "_input", queue_stats);
    }

    auto frame_opt = input_->Pop(std::chrono::milliseconds(10));
    if (!frame_opt)
    {
      return;
    }

    const data::Frame &frame = *frame_opt;

    int64_t now = context_->NowNanoseconds();
    double frame_age_ms = (now - frame.timestamp_ns) / 1e6;
    core::MetricsCollector::Instance().RecordFrameAge(get_name(), frame_age_ms);

    const data::TensorView &image = frame.image;
    const data::TensorShape &shape = image.shape();

    if (shape.rank() != 3)
    {
      context_->LogError("Invalid image shape");
      return;
    }

    core::MetricsCollector::Instance().IncrementFramesProcessed(get_name());
  }

} // namespace ptk::components

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::FrameDebugger)
