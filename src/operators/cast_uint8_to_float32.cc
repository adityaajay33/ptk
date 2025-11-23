#include "operators/cast_uint8_to_float32.h"

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"
#include <cstdint>

namespace ptk
{
    namespace operators
    {
        core::Status CastUint8ToFloat32(const data::TensorView &src, data::TensorView *dst)
        {
            if (dst == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "CastUint8ToFloat32: dst is null");
            }
            if (src.dtype() != core::DataType::kUint8 ||
                dst->dtype() != core::DataType::kFloat32)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "CastUint8ToFloat32: invalid data types");
            }

            if (src.shape().num_elements() != dst->shape().num_elements())
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "CastUint8ToFloat32: shape mismatch");
            }

            const std::uint8_t *in =
                static_cast<const std::uint8_t *>(src.buffer().data());
            float *out =
                static_cast<float *>(dst->buffer().data());

            std::size_t n =
                static_cast<std::size_t>(src.shape().num_elements());
            for (std::size_t i = 0; i < n; ++i)
            {
                out[i] = static_cast<float>(in[i]);
            }

            return core::Status::Ok();
        }
    }
}

