#ifndef OPERATORS_CHW_TO_HWC_H_
#define OPERATORS_CHW_TO_HWC_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        Status ChwToHwc(const TensorView &src, TensorView *dst);
    }
}

#endif // OPERATORS_CHW_TO_HWC_H_

