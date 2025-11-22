#include "runtime/components/heartbeat.h"

#include <string>  // for std::to_string

#include "runtime/core/runtime_context.h"

namespace runtime {

Heartbeat::Heartbeat() : context_(nullptr), count_(0) {}

Status Heartbeat::Init(RuntimeContext* context) {
  if (context == nullptr) {
    return Status(StatusCode::kInvalidArgument, "Context is null");
  }
  context_ = context;
  return Status::Ok();
}

Status Heartbeat::Start() {
  count_ = 0;
  context_->LogInfo("Heartbeat started.");
  return Status::Ok();
}

void Heartbeat::Stop() {
  std::string msg =
      "Heartbeat stopped at count: " + std::to_string(count_);
  context_->LogInfo(msg.c_str());
}

void Heartbeat::Tick() {
  ++count_;
  if (count_ % 5 == 0) {
    std::string msg =
        "Heartbeat tick: " + std::to_string(count_);
    context_->LogInfo(msg.c_str());
  }
}

}  // namespace runtime