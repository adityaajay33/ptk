#ifndef OPERATORS_CAST_FLOAT32_TO_UINT8_H_
#define OPERATORS_CAST_FLOAT32_TO_UINT8_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status CastFloat32ToUint8(const data::TensorView &src, data::TensorView *dst);
    }
}

#endif // OPERATORS_CAST_FLOAT32_TO_UINT8_H_

