#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>
#include <memory>

namespace ptk::core{

    enum class QueuePolicy {
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
    class BoundedQueue {
        public:
            BoundedQueue(size_t capacity, QueuePolicy policy) : capacity_(capacity), policy_(policy), running_(true) {}
            ~BoundedQueue(){
                Shutdown();
            }

            bool TryPush(T&& item){
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
                        queue_.pop();
                        stats_.total_dropped++;
                    }
                    queue_.push(std::move(item));
                    cv_.notify_one();
                    stats_.current_depth = queue_.size();
                    return true;
                }
                else{
                    return false;
                }
            }
            
            //only valid for kBlock - blocking push
            bool Push(T&& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(0)){
                std::unique_lock<std::mutex> lock(mutex_);

                if (policy_ == QueuePolicy::kBlock)
                {
                    return TryPush(std::move(item));
                }

                stats_.total_pushed++;

                if (timeout.count() > 0)
                {
                    if (!cv_producer_.wait_for(lock, timeout, [this] { return queue_.size() < capacity_ || !running_ }))
                    {
                        stats_.total_dropped++;
                        return false;
                    }
                }
                
            };

    };
}