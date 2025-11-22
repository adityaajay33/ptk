// src/components/frame_debugger.cc
#include "runtime/components/frame_debugger.h"

#include <cstdint>
#include <string>

#include "runtime/core/runtime_context.h"
#include "runtime/data/frame.h"
#include "runtime/data/tensor.h"  // whatever defines TensorView / TensorShape

namespace ptk {

FrameDebugger::FrameDebugger()
    : context_(nullptr), input_(nullptr), tick_count_(0) {}

void FrameDebugger::BindInput(InputPort<Frame>* port) {
  input_ = port;
}

Status FrameDebugger::Init(RuntimeContext* context) {
  if (context == nullptr) {
    return Status(StatusCode::kInvalidArgument, "Context is null");
  }
  context_ = context;
  tick_count_ = 0;
  return Status::Ok();
}

Status FrameDebugger::Start() {
  if (input_ == nullptr || !input_->is_bound()) {
    return Status(StatusCode::kFailedPrecondition,
                  "FrameDebugger input not bound");
  }
  context_->LogInfo("FrameDebugger started.");
  return Status::Ok();
}

void FrameDebugger::Stop() {
  std::string msg = "FrameDebugger stopped after " +
                    std::to_string(tick_count_) + " ticks.";
  context_->LogInfo(msg.c_str());
}

void FrameDebugger::Tick() {
  ++tick_count_;

  if (input_ == nullptr || !input_->is_bound()) {
    context_->LogError("FrameDebugger Tick with unbound input.");
    return;
  }

  const Frame* frame = input_->get();
  if (frame == nullptr) {
    context_->LogError("FrameDebugger Tick with null frame.");
    return;
  }

  // Direct member access, since Frame is a struct.
  const TensorView& image = frame->image;
  const TensorShape& shape = image.shape();

  if (shape.rank() != 3) {
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

}  // namespace ptk