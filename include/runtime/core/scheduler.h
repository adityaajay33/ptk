#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>

#include "runtime/components/component_interface.h"
#include "runtime/core/status.h"

namespace ptk::core
{
    class RuntimeContext;

    class Scheduler
    {
    public:
        Scheduler();
        ~Scheduler();

        Status Init(RuntimeContext *context);
        Status AddComponent(components::ComponentInterface *component);
        Status Start();
        void Stop();
        void RunLoop();

        // Mutex management for shared data instances
        std::mutex& GetDataMutex(void* data_ptr);

    private:
        RuntimeContext *context_;
        std::vector<components::ComponentInterface *> components_;
        std::vector<std::thread> threads_;
        std::atomic<bool> running_;

        // Registry of mutexes for shared pointers
        std::mutex registry_mutex_;
        std::map<void*, std::unique_ptr<std::mutex>> data_mutexes_;
    };
} // namespace ptk::core