#pragma once

// TensorRT is only available on Linux/Windows, not macOS
#ifdef PTK_ENABLE_CUDA

#include <NvInfer.h>
#include "runtime/core/types.h"

namespace ptk::trt
{

    inline nvinfer1::DataType TrtTypeFromPtk(core::DataType t)
    {
        switch (t)
        {
        case core::DataType::kFloat32:
            return nvinfer1::DataType::kFLOAT;
        case core::DataType::kInt32:
            return nvinfer1::DataType::kINT32;
        case core::DataType::kInt8:
            return nvinfer1::DataType::kINT8;
        case core::DataType::kFloat64:
            // TensorRT does not support Float64, assert unsupported
            assert(false && "TensorRT does not support DataType::kFloat64");
            return nvinfer1::DataType::kFLOAT; // unreachable
        default:
            assert(false && "Unsupported data type for TensorRT conversion");
            return nvinfer1::DataType::kFLOAT;
        }
    }

    inline std::size_t TrtElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        default:
            return 0;
        }
    }

} // namespace ptk::trt

#endif // PTK_ENABLE_CUDA