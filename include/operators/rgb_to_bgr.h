#ifndef OPERATORS_RGB_TO_BGR_H_
#define OPERATORS_RGB_TO_BGR_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        Status RgbToBgr(TensorView *tensor);
    }
}

#endif // OPERATORS_RGB_TO_BGR_H_

