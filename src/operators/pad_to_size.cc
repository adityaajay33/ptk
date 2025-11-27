#include "operators/pad_to_size.h"

#include "runtime/core/status.h"

namespace ptk::operators
{
        // TODO: Implement PadToSize
        core::Status PadToSize(const data::TensorView &src, int target_h, int target_w, data::TensorView *dst)
        {
            return core::Status(core::StatusCode::kInternal, "PadToSize not yet implemented");
        }
} // namespace ptk::operators

