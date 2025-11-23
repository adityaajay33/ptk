#ifndef OPERATORS_PAD_TO_SIZE_H_
#define OPERATORS_PAD_TO_SIZE_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status PadToSize(const data::TensorView &src, int target_h, int target_w, data::TensorView *dst);
    }
}

#endif // OPERATORS_PAD_TO_SIZE_H_

