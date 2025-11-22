#include "operators/pad_to_size.h"

#include "runtime/core/status.h"

namespace ptk
{
    namespace operators
    {
        // TODO: Implement PadToSize
        Status PadToSize(const TensorView &src, int target_h, int target_w, TensorView *dst)
        {
            return Status(StatusCode::kInternal, "PadToSize not yet implemented");
        }
    }
}

