#include "runtime/core/pipeline.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"

namespace runtime {

    Pipeline::Pipeline(): context_(nullptr), scheduler_(nullptr), built_(false), components_() {}

    Pipeline::~Pipeline() = default;

    Status Pipeline::Build(RuntimeContext* context) {
        if (context==nullptr) {
            return Status(StatusCode::kInvalidArgument, "Context is null");
        }
        if (built_) {

            return Status(StatusCode::kFailedPrecondition, "Pipeline::Build() called more than once");
        }

        Status s = DoBuild(context);
        if (!s.ok()) {
            return s;
        }

        built_ = true;
        return Status::Ok();
    }

    Status Pipeline::RegisterComponents(Scheduler* scheduler) {
        if (scheduler == nullptr) {
            return Status(StatusCode::kInvalidArgument, "Scheduler is null");
        }
        if (!built_)
        {
            return Status(StatusCode::kFailedPrecondition, "Pipeline::Build() must be called first");
        }

        for (auto* component : components_) {
            Status s = scheduler->AddComponent(component);
            if (!s.ok()) {
                return s;
            }
        }

        return Status::Ok();
        
    }

    Status Pipeline::AddComponent(ComponentInterface* component){
        if (component == nullptr){
            return Status(StatusCode::kInvalidArgument, "Component is null");
        }

        components_.push_back(component);
        return Status::Ok();
    };
} // namespace runtime