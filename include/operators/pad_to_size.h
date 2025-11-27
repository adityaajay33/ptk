#pragma once

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk::operators
{
    core::Status PadToSize(const data::TensorView &src, int target_h, int target_w, data::TensorView *dst);
}