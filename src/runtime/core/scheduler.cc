#include "runtime/core/scheduler.h"
#include "runtime/core/runtime_context.h"
#include <chrono>

namespace ptk::core
{
    Scheduler::Scheduler() : context_(nullptr), components_(), running_(false) {}

    Scheduler::~Scheduler()
    {
        Stop();
    }

    Status Scheduler::Init(RuntimeContext *context)
    {
        if (context == nullptr) return Status(StatusCode::kInvalidArgument, "Context is null");
        context_ = context;
        return Status::Ok();
    }

    Status Scheduler::AddComponent(components::ComponentInterface *component)
    {
        if (component == nullptr) return Status(StatusCode::kInvalidArgument, "Component is null");
        component->SetScheduler(this);
        components_.push_back(component);
        return Status::Ok();
    }

    Status Scheduler::Start()
    {
        if (running_) return Status(StatusCode::kFailedPrecondition, "Scheduler is already running");

        for (auto *c : components_)
        {
            Status s = c->Init(context_);
            if (!s.ok()) return s;
            s = c->Start();
            if (!s.ok()) return s;
        }

        running_ = true;
        for (auto *c : components_)
        {
            threads_.emplace_back([this, c]() {
                while (running_)
                {
                    c->Tick();
                    // Basic pacing to prevent 100% CPU in simple source components
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
        }

        return Status::Ok();
    }

    void Scheduler::Stop()
    {
        if (!running_) return;

        running_ = false;
        for (auto &t : threads_)
        {
            if (t.joinable()) t.join();
        }
        threads_.clear();

        for (auto *c : components_)
        {
            c->Stop();
        }
        
        std::lock_guard<std::mutex> lock(registry_mutex_);
        data_mutexes_.clear();
    }

    void Scheduler::RunLoop()
    {
        while (running_)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::mutex& Scheduler::GetDataMutex(void* data_ptr)
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = data_mutexes_.find(data_ptr);
        if (it == data_mutexes_.end())
        {
            it = data_mutexes_.emplace(data_ptr, std::make_unique<std::mutex>()).first;
        }
        return *(it->second);
    }

} // namespace ptk::core