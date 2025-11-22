#include "operators/center_crop.h"

#include "runtime/core/status.h"

namespace ptk
{
    namespace operators
    {
        // TODO: Implement CenterCrop
        Status CenterCrop(const TensorView &src, int crop_h, int crop_w, TensorView *dst)
        {
            return Status(StatusCode::kInternal, "CenterCrop not yet implemented");
        }
    }
}

