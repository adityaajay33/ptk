#include "operators/add_batch_dim.h"

#include "runtime/core/status.h"

namespace ptk
{
    namespace operators
    {
        // TODO: Implement AddBatchDim
        Status AddBatchDim(const TensorView &src, TensorView *dst)
        {
            return Status(StatusCode::kInternal, "AddBatchDim not yet implemented");
        }
    }
}

