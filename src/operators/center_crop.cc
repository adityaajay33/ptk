#include "operators/center_crop.h"

#include "runtime/core/status.h"

namespace ptk
{
    namespace operators
    {
        // TODO: Implement CenterCrop
        core::Status CenterCrop(const data::TensorView &src, int crop_h, int crop_w, data::TensorView *dst)
        {
            return core::Status(core::StatusCode::kInternal, "CenterCrop not yet implemented");
        }
    }
}

