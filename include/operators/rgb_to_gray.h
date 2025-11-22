#ifndef OPERATORS_RGB_TO_GRAY_H_
#define OPERATORS_RGB_TO_GRAY_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        Status RgbToGray(const TensorView &src, TensorView *dst);
    }
}

#endif // OPERATORS_RGB_TO_GRAY_H_

