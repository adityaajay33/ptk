#pragma once

#include "operators/normalization_params.h"
#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status Normalize(data::TensorView *tensor, const NormalizationParams &params, core::TensorLayout layout);
    }
}