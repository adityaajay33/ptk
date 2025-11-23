#ifndef OPERATORS_ADD_BATCH_DIM_H_
#define OPERATORS_ADD_BATCH_DIM_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status AddBatchDim(const data::TensorView &src, data::TensorView *dst);
    }
}

#endif // OPERATORS_ADD_BATCH_DIM_H_

