#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>
#include <memory>

namespace ptk::core
{

    enum class QueuePolicy
    {
        kLatestOnly,
        kDropOldest,
        kBlock,
        kDropNewest,
    };

    struct QueueStats
    {
        size_t total_pushed = 0;
        size_t total_popped = 0;
        size_t total_dropped = 0;
        size_t current_depth = 0;
        double max_age_ms = 0.0;
    };

    template <typename T>
    class BoundedQueue
    {
    public:
        BoundedQueue(size_t capacity, QueuePolicy policy) : capacity_(capacity), policy_(policy), running_(true) {}
        ~BoundedQueue()
        {
            Shutdown();
        }

        // non blocking
        bool TryPush(T &&item)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stats_.total_pushed++;

            if (policy_ == QueuePolicy::kLatestOnly)
            {
                if (!queue_.empty())
                {
                    queue_.pop();
                    stats_.total_dropped++;
                }

                queue_.push(std::move(item));
                cv_.notify_one();
                stats_.current_depth = queue_.size();
                return true;
            }
            else if (policy_ == QueuePolicy::kDropOldest)
            {
                if (queue_.size() >= capacity_)
                {
                    queue_.pop();
                    stats_.total_dropped++;
                }
                queue_.push(std::move(item));
                cv_.notify_one();
                stats_.current_depth = queue_.size();
                return true;
            }
            else if (policy_ == QueuePolicy::kDropNewest)
            {
                if (queue_.size() >= capacity_)
                {
                    stats_.total_dropped++;
                    return false;  // Reject new item when full
                }
                queue_.push(std::move(item));
                cv_.notify_one();
                stats_.current_depth = queue_.size();
                return true;
            }
            else
            {
                return false;
            }
        }

        // only valid for kBlock - blocking push
        bool Push(T &&item, std::chrono::milliseconds timeout = std::chrono::milliseconds(0))
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (policy_ != QueuePolicy::kBlock)
            {
                lock.unlock();
                return TryPush(std::move(item));
            }

            stats_.total_pushed++;

            if (timeout.count() > 0)
            {
                if (!cv_producer_.wait_for(lock, timeout, [this]
                                           { return queue_.size() < capacity_ || !running_ }))
                {
                    stats_.total_dropped++;
                    return false;
                }
            }
            else
            {
                cv_producer_.wait(lock, [this]
                                  { return queue_.size() < capacity_ || !running_ });
            }

            if (!running_)
            {
                return false;
            }

            queue_.push(std::move(item));
            cv_.notify_one();
            stats_.current_depth = queue_.size();
            return true;
        }

        // non blocking
        std::optional<T> TryPop()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.empty())
            {
                return std::nullopt;
            }

            T item = std::move(queue_.front());
            queue_.pop();
            stats_.total_popped++;
            stats_.current_depth = queue_.size();

            if (policy_ == QueuePolicy::kBlock)
            {
                cv_producer_.notify_one();
            }

            return item;
        }

        std::optional<T> Pop(std::chrono::milliseconds timeout = std::chrono::milliseconds(0))
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (timeout.count() > 0)
            {
                if (!cv_.wait_for(lock, timeout, [this]
                                  { return !queue_.empty() || !running_; }))
                {
                    return std::nullopt;  // Timeout
                }
            }
            else
            {
                cv_.wait(lock, [this]
                         { return !queue_.empty() || !running_; });
            }

            if (queue_.empty() || !running_)
            {
                return std::nullopt;
            }

            T item = std::move(queue_.front());
            queue_.pop();
            stats_.total_popped++;
            stats_.current_depth = queue_.size();

            if (policy_ == QueuePolicy::kBlock)
            {
                cv_producer_.notify_one();
            }

            return item;
        }

        size_t Size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

        bool Empty() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

        QueueStats GetStats() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return stats_;
        }

        void Clear()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!queue_.empty())
            {
                queue_.pop();
            }
            stats_.current_depth = 0;
        }

        void Shutdown()
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                running_ = false;
            } // extra braces means lock gets released before notify_all
            cv_.notify_all();
            cv_producer_.notify_all();
        }

    private:
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::condition_variable cv_producer_;
        std::queue<T> queue_;
        size_t capacity_;
        QueuePolicy policy_;
        QueueStats stats_;
        bool running_;
    };

}  // namespace ptk::core
