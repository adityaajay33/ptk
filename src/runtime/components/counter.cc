#include "runtime/components/counter.h"
#include "runtime/core/runtime_context.h"

namespace ptk
{
    namespace components
    {

        Counter::Counter() : context_(nullptr), count_(0) {}

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
            context_->LogInfo("Counter started.");
            return core::Status::Ok();
        }

        core::Status Counter::Stop()
        {
            context_->LogInfo("Counter stopped at count: " + std::to_string(count_));
            return core::Status::Ok();
        }

        void Counter::Tick()
        {
            ++count_;
            if (count_ % 10 == 0)
            {
                const std::string msg =
                    "Counter reached " + std::to_string(count_) + " ticks.";
                context_->LogInfo(msg.c_str());
            }
        }

    } // namespace components
} // namespace ptk