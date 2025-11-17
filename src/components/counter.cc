#include "runtime/components/counter.h"
#include "runtime/core/runtime_context.h"

namespace runtime {

    Counter::Counter() : context_(nullptr), count_(0) {}

    Status Counter::Init(RuntimeContext* context) {
        if (context == nullptr) {
            return Status(StatusCode::kInvalidArgument, "Context is null");
        }
        context_ = context;
        return Status::Ok();
    }

    Status Counter::Start(){
        count_ = 0;
        context_ -> LogInfo("Counter started.");
        return Status::Ok();
    }

    void Counter::Stop() {
        context_->LogInfo("Counter stopped at count: " + std::to_string(count_));
    }

    void Counter::Tick() {
    ++count_;
    if (count_ % 10 == 0) {
        const std::string msg =
            "Counter reached " + std::to_string(count_) + " ticks.";
        context_->LogInfo(msg.c_str());
    }
    }
} // namespace runtime