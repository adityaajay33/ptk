#include "operators/add_batch_dim.h"

#include "runtime/core/status.h"

namespace ptk
{
    namespace operators
    {
        // TODO: Implement AddBatchDim
        core::Status AddBatchDim(const data::TensorView &src, data::TensorView *dst)
        {
            return core::Status(core::StatusCode::kInternal, "AddBatchDim not yet implemented");
        }
    }
}

