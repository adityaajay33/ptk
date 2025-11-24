#pragma once

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status BgrToRgb(data::TensorView *tensor);
    }
}