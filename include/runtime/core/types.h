#ifndef RUNTIME_CORE_TYPES_H_
#define RUNTIME_CORE_TYPES_H_

#include <cstdint>


namespace ptk {
    enum class DeviceType {
        kCpu = 0,
        kCuda,
    };

    enum class DataType {
        kUnknown = 0,
        kUint8,
        kInt32,
        kInt64, 
        kFloat32,
        kFloat64,
    };

    enum class TensorLayout {
        kUnknown = 0,
        kHwc,  // height, width, channels
        kChw,  // channels, height, width
        kNhwc, // batch, height, width, channels
        kNchw, // batch, channels, height, width

    };

    enum class PixelFormat {

        kUnknown = 0,
        kGray8,  // 1 channel
        kRgb8,   // 3 channels
        kBgr8,   // 3 channels
        kRgba8,  // 4 channels
    };

}

#endif // RUNTIME_CORE_TYPES_H_