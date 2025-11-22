#ifndef OPERATORS_BGR_TO_RGB_H_
#define OPERATORS_BGR_TO_RGB_H_

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        Status BgrToRgb(TensorView *tensor);
    }
}

#endif // OPERATORS_BGR_TO_RGB_H_

