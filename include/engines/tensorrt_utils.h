#pragma once

// TensorRT is only available on Linux/Windows, not macOS
#ifdef __APPLE__
#warning "TensorRT is not supported on macOS. This file will be empty on macOS builds."
#else

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
            return nvinfer1::DataType::kFLOAT;
        default:
            return nvinfer1::DataType::kFLOAT; // fallback, TRT doesn't support many types
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

#endif // !__APPLE__