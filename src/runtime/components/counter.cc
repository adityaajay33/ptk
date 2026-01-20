#include "runtime/components/counter.h"
#include "runtime/core/runtime_context.h"

namespace ptk::components
{

        Counter::Counter(const rclcpp::NodeOptions &options)
            : ComponentInterface("counter", options),
              context_(nullptr),
              count_(0) {}

        core::Status Counter::Init(core::RuntimeContext *context)
        {
            if (context == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument, "Context is null");
            }
            context_ = context;
            return core::Status::Ok();
        }

        core::Status Counter::Start()
        {
            count_ = 0;
            return core::Status::Ok();
        }

        core::Status Counter::Stop()
        {
            return core::Status::Ok();
        }

        void Counter::Tick()
        {
            ++count_;
        }

} // namespace ptk::components

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::Counter)