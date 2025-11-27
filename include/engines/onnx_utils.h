#pragma once

#include <onnxruntime_cxx_api.h>
#include "runtime/core/types.h"

namespace ptk::onnx
{
    inline ONNXTensorElementDataType OnnxTypeFromPtkType(core::DataType t)
    {
        switch (t)
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

    inline core::DataType PtkTypeFromOnnx(ONNXTensorElementDataType t)
    {
        switch (t)
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
}