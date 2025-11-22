#ifndef OPERATORS_NORMALIZE_H_
#define OPERATORS_NORMALIZE_H_

#include "operators/normalization_params.h"
#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        Status Normalize(TensorView *tensor, const NormalizationParams &params, TensorLayout layout);
    }
}

#endif // OPERATORS_NORMALIZE_H_

