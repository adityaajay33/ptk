#pragma once

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status CenterCrop(const data::TensorView &src, int crop_h, int crop_w, data::TensorView *dst);
    }
}