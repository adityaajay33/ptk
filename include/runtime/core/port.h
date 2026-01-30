#pragma once

#include "runtime/core/queue_policy.h"
#include <atomic>

namespace ptk::core
{

        template <typename T>
        class OutputPort
        {
        public:
            OutputPort() : queue_(nullptr) {}

            void Bind(std::shared_ptr<BoundedQueue<T>> queue)
            {
                queue_ = queue;
            }

            bool Push(T&& item)
            {
                if (!queue_) return false;
                return queue_->TryPush(std::move(item));
            }

            bool is_bound() const
            {
                return queue_ != nullptr;
            }

            QueueStats GetStats() const
            {
                if (!queue_) return QueueStats();
                return queue_->GetStats();
            }

        private:
            std::shared_ptr<BoundedQueue<T>> queue_;
        };

        template <typename T>
        class InputPort
        {
        public:
            InputPort() : queue_(nullptr) {}

            void Bind(std::shared_ptr<BoundedQueue<T>> queue)
            {
                queue_ = queue;
            }

            bool is_bound() const
            {
                return queue_ != nullptr;
            }

            std::optional<T> Pop(std::chrono::milliseconds timeout = std::chrono::milliseconds(0))
            {
                if (!queue_) return std::nullopt;
                return queue_->Pop(timeout);
            }

            std::optional<T> TryPop()
            {
                if (!queue_) return std::nullopt;
                return queue_->TryPop();
            }
            QueueStats GetStats() const
            {
                if (!queue_) return QueueStats();
                return queue_->GetStats();
            }

        private:
            std::shared_ptr<BoundedQueue<T>> queue_;
        };

}  // namespace ptk::core