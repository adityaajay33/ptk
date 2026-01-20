#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk::engine
{

    struct TensorMetadata
    {
        std::string name;          
        size_t binding_index = 0;     
        bool is_input = false;        
        bool is_output = false;       
        bool is_dynamic_shape = false; 
        std::vector<size_t> min_dims;  
        std::vector<size_t> max_dims;  
    };

    class EngineTensor
    {
    public:
        EngineTensor() = default;

        explicit EngineTensor(const data::TensorView &tensor_view);

        EngineTensor(const data::TensorView &tensor_view,
                     const TensorMetadata &metadata);

        EngineTensor(const EngineTensor &) = default;
        EngineTensor &operator=(const EngineTensor &) = default;
        EngineTensor(EngineTensor &&) = default;
        EngineTensor &operator=(EngineTensor &&) = default;

        ~EngineTensor() = default;

        const data::TensorView &GetTensorView() const { return tensor_view_; }
        data::TensorView &GetTensorView() { return tensor_view_; }

        const TensorMetadata &GetMetadata() const { return metadata_; }
        TensorMetadata &GetMetadata() { return metadata_; }

        const std::string &GetName() const { return metadata_.name; }
        core::DataType GetDataType() const { return tensor_view_.dtype(); }
        const data::TensorShape &GetShape() const { return tensor_view_.shape(); }
        size_t GetNumElements() const { return tensor_view_.num_elements(); }
        size_t GetSizeBytes() const { return tensor_view_.buffer().size_bytes(); }
        core::DeviceType GetDeviceType() const
        {
            return tensor_view_.buffer().device_type();
        }

        void *GetData() { return tensor_view_.buffer().data(); }
        const void *GetData() const { return tensor_view_.buffer().data(); }

        bool IsInput() const { return metadata_.is_input; }
        bool IsOutput() const { return metadata_.is_output; }
        bool HasDynamicShape() const { return metadata_.is_dynamic_shape; }
        size_t GetBindingIndex() const { return metadata_.binding_index; }

        core::Status ValidateShape(const data::TensorShape &expected_shape) const;
        core::Status ValidateShape(const std::vector<int64_t> &expected_dims) const;

        core::Status SetDynamicShapeBounds(const std::vector<size_t> &min_dims,
                                           const std::vector<size_t> &max_dims);

        core::Status ValidateDynamicShape() const;

        core::Status ValidateDataType(core::DataType expected_dtype) const;

        core::Status ValidateDevice(core::DeviceType expected_device) const;

        core::Status Validate() const;

        std::string ToString() const;

    private:
        data::TensorView tensor_view_;
        TensorMetadata metadata_;
    };

    namespace type_mapping
    {
#ifndef __APPLE__
#include <onnxruntime_cxx_api.h>

        inline ONNXTensorElementDataType PtkToOnnxType(core::DataType dtype)
        {
            switch (dtype)
            {
            case core::DataType::kUint8:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            case core::DataType::kInt32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            case core::DataType::kInt64:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            case core::DataType::kFloat32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            case core::DataType::kFloat64:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
            default:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            }
        }

        inline core::DataType OnnxToPtkType(ONNXTensorElementDataType dtype)
        {
            switch (dtype)
            {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                return core::DataType::kUint8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                return core::DataType::kInt32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                return core::DataType::kInt64;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                return core::DataType::kFloat32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                return core::DataType::kFloat64;
            default:
                return core::DataType::kUnknown;
            }
        }

#endif
#ifndef __APPLE__
#include <NvInfer.h>

        inline nvinfer1::DataType PtkToTrtType(core::DataType dtype)
        {
            switch (dtype)
            {
            case core::DataType::kFloat32:
                return nvinfer1::DataType::kFLOAT;
            case core::DataType::kFloat64:
                // TensorRT doesn't support float64
                return nvinfer1::DataType::kFLOAT;
            case core::DataType::kInt32:
                return nvinfer1::DataType::kINT32;
            case core::DataType::kInt64:
                return nvinfer1::DataType::kINT32; // Cast to INT32
            case core::DataType::kUint8:
                return nvinfer1::DataType::kINT8;
            default:
                return nvinfer1::DataType::kFLOAT;
            }
        }

        inline core::DataType TrtToPtkType(nvinfer1::DataType dtype)
        {
            switch (dtype)
            {
            case nvinfer1::DataType::kFLOAT:
                return core::DataType::kFloat32;
            case nvinfer1::DataType::kHALF:
                return core::DataType::kFloat32; // Downcast to float32
            case nvinfer1::DataType::kINT32:
                return core::DataType::kInt32;
            case nvinfer1::DataType::kINT8:
                return core::DataType::kUint8;
            case nvinfer1::DataType::kBOOL:
                return core::DataType::kUint8; // Represent as uint8
            default:
                return core::DataType::kUnknown;
            }
        }

#endif

    } // namespace type_mapping

    namespace shape_inference
    {
        class ShapeCalculator
        {
        public:
            static data::TensorShape ElementWise(const data::TensorShape &input);

            static data::TensorShape Reshape(const data::TensorShape &input,
                                             const std::vector<int64_t> &new_shape);

            static data::TensorShape Transpose(const data::TensorShape &input,
                                               const std::vector<int> &perm);

            static data::TensorShape Reduce(const data::TensorShape &input,
                                            const std::vector<int> &axes,
                                            bool keep_dims = false);

            static core::Status Concatenate(const std::vector<data::TensorShape> &inputs,
                                            int axis,
                                            data::TensorShape *output);

            static core::Status MatMul(const data::TensorShape &lhs,
                                       const data::TensorShape &rhs,
                                       data::TensorShape *output);

            static core::Status BroadcastBinary(const data::TensorShape &lhs,
                                                const data::TensorShape &rhs,
                                                data::TensorShape *output);

            static data::TensorShape Flatten(const data::TensorShape &input,
                                             int start_axis = 0,
                                             int end_axis = -1);

            static core::Status Squeeze(const data::TensorShape &input,
                                        const std::vector<int> &axes,
                                        data::TensorShape *output);

            static core::Status Unsqueeze(const data::TensorShape &input,
                                          const std::vector<int> &axes,
                                          data::TensorShape *output);

            static data::TensorShape Slice(const data::TensorShape &input,
                                           const std::vector<int> &starts,
                                           const std::vector<int> &ends);

            static core::Status BroadcastShapes(const data::TensorShape &shape1,
                                                const data::TensorShape &shape2,
                                                data::TensorShape *output);
        };

        inline int GetBatchSize(const data::TensorShape &shape)
        {
            if (shape.rank() > 0)
            {
                return shape.dim(0);
            }
            return 1;
        }

        inline int SetBatchSize(const data::TensorShape &shape, int new_batch_size,
                                data::TensorShape *output)
        {
            auto dims = shape.dims();
            if (dims.empty())
            {
                return -1;
            }
            dims[0] = new_batch_size;
            *output = data::TensorShape(dims);
            return 0;
        }

        inline bool IsScalar(const data::TensorShape &shape)
        {
            return shape.rank() == 0;
        }

        inline bool IsVector(const data::TensorShape &shape)
        {
            return shape.rank() == 1;
        }

        inline bool IsMatrix(const data::TensorShape &shape)
        {
            return shape.rank() == 2;
        }

        inline bool Is3D(const data::TensorShape &shape)
        {
            return shape.rank() == 3;
        }

        inline bool Is4D(const data::TensorShape &shape)
        {
            return shape.rank() == 4;
        }

    } // namespace shape_inference

    class EngineTensorBatch
    {
    public:
        explicit EngineTensorBatch(size_t capacity = 32)
        {
            tensors_.reserve(capacity);
        }

        void Add(const EngineTensor &tensor) { tensors_.push_back(tensor); }

        EngineTensor &Get(size_t index) { return tensors_[index]; }
        const EngineTensor &Get(size_t index) const { return tensors_[index]; }

        size_t Size() const { return tensors_.size(); }
        bool Empty() const { return tensors_.empty(); }

        void Clear() { tensors_.clear(); }

        core::Status ValidateAll() const;

        std::vector<std::string> GetNames() const;

        std::vector<core::DataType> GetDataTypes() const;

    private:
        std::vector<EngineTensor> tensors_;
    };

} // namespace ptk::engine
