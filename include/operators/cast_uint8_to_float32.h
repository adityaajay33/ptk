#pragma once

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status CastUint8ToFloat32(const data::TensorView &src, data::TensorView *dst);
    }
}