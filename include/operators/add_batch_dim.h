#pragma once

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk::operators
{
    core::Status AddBatchDim(const data::TensorView &src, data::TensorView *dst);
}
