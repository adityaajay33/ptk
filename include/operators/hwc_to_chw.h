#ifndef OPERATORS_HWC_TO_CHW_H_
#define OPERATORS_HWC_TO_CHW_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status HwcToChw(const data::TensorView &src, data::TensorView *dst);
    }
}

#endif // OPERATORS_HWC_TO_CHW_H_

