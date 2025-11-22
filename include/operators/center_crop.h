#ifndef OPERATORS_CENTER_CROP_H_
#define OPERATORS_CENTER_CROP_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        Status CenterCrop(const TensorView &src, int crop_h, int crop_w, TensorView *dst);
    }
}

#endif // OPERATORS_CENTER_CROP_H_

