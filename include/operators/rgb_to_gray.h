#ifndef OPERATORS_RGB_TO_GRAY_H_
#define OPERATORS_RGB_TO_GRAY_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status RgbToGray(const data::TensorView &src, data::TensorView *dst);
    }
}

#endif // OPERATORS_RGB_TO_GRAY_H_

