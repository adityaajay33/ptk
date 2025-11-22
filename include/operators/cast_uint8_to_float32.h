#ifndef OPERATORS_CAST_UINT8_TO_FLOAT32_H_
#define OPERATORS_CAST_UINT8_TO_FLOAT32_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        Status CastUint8ToFloat32(const TensorView &src, TensorView *dst);
    }
}

#endif // OPERATORS_CAST_UINT8_TO_FLOAT32_H_

