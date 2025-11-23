#include "runtime/components/heartbeat.h"

#include <string>  // for std::to_string

#include "runtime/core/runtime_context.h"

namespace ptk {
namespace components {

Heartbeat::Heartbeat() : context_(nullptr), count_(0) {}

core::Status Heartbeat::Init(core::RuntimeContext* context) {
  if (context == nullptr) {
    return core::Status(core::StatusCode::kInvalidArgument, "Context is null");
  }
  context_ = context;
  return core::Status::Ok();
}

core::Status Heartbeat::Start() {
  count_ = 0;
  context_->LogInfo("Heartbeat started.");
  return core::Status::Ok();
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

}  // namespace components
}  // namespace ptk