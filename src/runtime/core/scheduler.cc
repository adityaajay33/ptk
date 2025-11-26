#include "runtime/core/scheduler.h"
#include "runtime/core/runtime_context.h"

namespace ptk
{
    namespace core
    {

        Scheduler::Scheduler() : context_(nullptr), components_(), running_(false) {}

        Status Scheduler::Init(RuntimeContext *context)
        {
            if (context == nullptr)
            {
                return Status(StatusCode::kInvalidArgument, "Context is null");
            }
            if (context_ != nullptr)
            {
                return Status(StatusCode::kFailedPrecondition, "Scheduler::Init() called more than once");
            }
            context_ = context;
            return Status::Ok();
        }

        Status Scheduler::AddComponent(components::ComponentInterface *component)
        {
            if (!context_)
            {
                return Status(StatusCode::kFailedPrecondition,
                              "Scheduler::Init must be called before AddComponent");
            }
            if (component == nullptr)
            {
                return Status(StatusCode::kInvalidArgument, "Component is null");
            }
            components_.push_back(component);
            return Status::Ok();
        }

        Status Scheduler::Start()
        {

            if (!context_)
            {
                return Status(StatusCode::kFailedPrecondition,
                              "Scheduler::Init must be called before Start");
            }
            if (running_)
            {
                return Status(StatusCode::kFailedPrecondition, "Scheduler is already running");
            }
            if (components_.empty())
            {
                return Status(StatusCode::kFailedPrecondition, "No components to run");
            }

            for (auto *c : components_)
            {
                Status s = c->Init(context_);
                if (!s.ok())
                {
                    return s;
                }
                s = c->Start();
                if (!s.ok())
                {
                    return s;
                }
            }

            tick_ = 0;
            running_ = true;
            return Status::Ok();
        }

        void Scheduler::Stop()
        {
            if (!running_)
            {
                return;
            }

            for (auto *c : components_)
            {
                c->Stop();
            }

            running_ = false;
        }

        void Scheduler::RunLoop(int num_ticks)
        {
            if (!running_)
            {
                return;
            }
            for (int i = 0; i < num_ticks && running_; ++i)
            {
                ++tick_;
                for (auto *c : components_)
                {
                    c->Tick();
                }
            }
        }

    } // namespace core
} // namespace ptk