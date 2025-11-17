#ifndef RUNTIME_DATA_TENSOR_H_
#define RUNTIME_DATA_TENSOR_H_

#include <cstdint>
#include <vector>

#include "runtime/core/types.h"
#include "runtime/data/buffer.h"

namespace runtime {

    class TensorShape {

        public:
            TensorShape() : dims_() {}

            explicit TensorShape(const std::vector<std::int64_t>& dims) : dims_(dims) {}

            const std::vector<std::int64_t>& dims() const { return dims_; }
            std::vector<int64_t>& dims() { return dims_; }

            std::size_t rank() const { return dims_.size(); }

            int64_t dim(std::size_t index) const { return dims_[index]; }

        private:

            std::vector<std::int64_t> dims_;


    };

    class TensorView {
        public:
            
            TensorView() : buffer_view_(), data_type_(DataType::kUnknown), shape_() {}

            TensorView(const BufferView& buffer_view, DataType data_type, const TensorShape& shape)
                : buffer_view_(buffer_view), data_type_(data_type), shape_(shape) {}

            const BufferView& buffer() const { return buffer_view_; }
            BufferView& buffer() { return buffer_view_; }

            DataType data_type() const { return data_type_; }

            const TensorShape& shape() const { return shape_; }
            TensorShape& shape() { return shape_; }

            DeviceType device_type() const { return buffer_view_.device_type(); }

            bool empty() const { return buffer_view_.empty() || data_type_ == DataType::kUnknown; }

        private:
            BufferView buffer_view_;
            DataType data_type_;
            TensorShape shape_;
    };
}

#endif // RUNTIME_DATA_TENSOR_H_