#pragma once

#include <cstdint>
#include <vector>

#include "runtime/core/types.h"
#include "runtime/data/buffer.h"

namespace ptk {
namespace data {

    class TensorShape {

        public:
            TensorShape() : dims_() {}

            explicit TensorShape(const std::vector<std::int64_t>& dims) : dims_(dims) {}

            const std::vector<std::int64_t>& dims() const { return dims_; }
            std::vector<int64_t>& dims() { return dims_; }

            std::size_t rank() const { return dims_.size(); }

            int64_t dim(std::size_t index) const { return dims_[index]; }

            std::int64_t num_elements() const {
                if (dims_.empty()) {
                    return 0;
                }
                std::int64_t total = 1;
                for (std::int64_t d : dims_) {
                    total *= d;
                }
                return total;
            }

        private:

            std::vector<std::int64_t> dims_;


    };

    class TensorView {
        public:
            
            TensorView() : buffer_view_(), data_type_(core::DataType::kUnknown), shape_() {}

            TensorView(const BufferView& buffer_view, core::DataType data_type, const TensorShape& shape)
                : buffer_view_(buffer_view), data_type_(data_type), shape_(shape) {}

            core::DataType dtype() const { return data_type_;}

            const BufferView& buffer() const { return buffer_view_; }
            BufferView& buffer() { return buffer_view_; }

            core::DataType data_type() const { return data_type_; }

            const TensorShape& shape() const { return shape_; }
            TensorShape& shape() { return shape_; }

            core::DeviceType device_type() const { return buffer_view_.device_type(); }

            bool empty() const { return buffer_view_.empty() || data_type_ == core::DataType::kUnknown; }

        private:
            BufferView buffer_view_;
            core::DataType data_type_;
            TensorShape shape_;
    };

} // namespace data
}