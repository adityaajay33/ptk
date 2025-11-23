#ifndef OPERATORS_CHW_TO_HWC_H_
#define OPERATORS_CHW_TO_HWC_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status ChwToHwc(const data::TensorView &src, data::TensorView *dst);
    }
}

#endif // OPERATORS_CHW_TO_HWC_H_

