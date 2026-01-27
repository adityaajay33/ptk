#pragma once

#include <atomic>

namespace ptk::core
{

        template <typename T>
        class OutputPort
        {
        public:
            OutputPort() : value_(nullptr) {}

            void Bind(T *value)
            {
                value_.store(value, std::memory_order_release);
            }

            bool is_bound() const
            {
                return value_.load(std::memory_order_acquire) != nullptr;
            }

            T *get() const
            {
                return value_.load(std::memory_order_acquire);
            }

        private:
            std::atomic<T *> value_;
        };

        template <typename T>
        class InputPort
        {
        public:
            InputPort() : value_(nullptr) {}

            void Bind(T *value)
            {
                value_.store(value, std::memory_order_release);
            }

            bool is_bound() const
            {
                return value_.load(std::memory_order_acquire) != nullptr;
            }

            const T *get() const
            {
                return value_.load(std::memory_order_acquire);
            }

        private:
            std::atomic<const T *> value_;
        };

}  // namespace ptk::core