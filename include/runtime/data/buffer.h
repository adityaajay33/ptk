#ifndef RUNTIME_DATA_BUFFER_H_
#define RUNTIME_DATA_BUFFER_H_

#include <cstddef>
#include <cstdint>

#include "runtime/core/types.h"

namespace ptk {

    class BufferView {
        public:
            BufferView() : data_(nullptr), size_bytes_(0),    device_type_(DeviceType::kCpu) {}

            BufferView(void* data, std::size_t size_bytes, DeviceType device_type)
                : data_(data), size_bytes_(size_bytes), device_type_(device_type) {}

            void* data() { return data_; }
            const void* data() const { return data_; }

            std::size_t size_bytes() const { return size_bytes_; }

            DeviceType device_type() const { return device_type_; }

            bool empty() const { return data_ == nullptr || size_bytes_ == 0; }

        private:
            void* data_;
            std::size_t size_bytes_;
            DeviceType device_type_;
    };

} // namespace ptk

#endif // RUNTIME_DATA_BUFFER_H_