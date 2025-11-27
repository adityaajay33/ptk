#pragma once

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk::operators
{
    core::Status RgbToBgr(const data::TensorView &src, data::TensorView *dst);
}