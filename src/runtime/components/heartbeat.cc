#include "runtime/components/heartbeat.h"

#include <string> // for std::to_string

#include "runtime/core/runtime_context.h"

namespace ptk::components
{

    Heartbeat::Heartbeat(const rclcpp::NodeOptions &options)
        : ComponentInterface("heartbeat", options),
          context_(nullptr),
          count_(0) {}

    core::Status Heartbeat::Init(core::RuntimeContext *context)
    {
      if (context == nullptr)
      {
        return core::Status(core::StatusCode::kInvalidArgument, "Context is null");
  }
  context_ = context;
      return core::Status::Ok();
}

    core::Status Heartbeat::Start()
    {
        count_ = 0;
        return core::Status::Ok();
    }

    core::Status Heartbeat::Stop()
    {
        return core::Status::Ok();
    }

    void Heartbeat::Tick()
    {
        ++count_;
    }

} // namespace ptk::components

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::Heartbeat)