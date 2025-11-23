#include "operators/bgr_to_rgb.h"
#include "operators/rgb_to_bgr.h"

#include "runtime/core/status.h"

namespace ptk
{
    namespace operators
    {
        core::Status BgrToRgb(data::TensorView *tensor)
        {
            // Same swap as RgbToBgr
            return RgbToBgr(tensor);
        }
    }
}

