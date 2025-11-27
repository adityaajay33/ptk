#include "operators/cast_float32_to_uint8.h"

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"
#include <cstdint>

namespace ptk::operators
{
        core::Status CastFloat32ToUint8(const data::TensorView &src, data::TensorView *dst)
        {
            if (dst == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "CastFloat32ToUint8: dst is null");
            }
            if (src.dtype() != core::DataType::kFloat32 ||
                dst->dtype() != core::DataType::kUint8)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "CastFloat32ToUint8: invalid data types");
            }

            if (src.shape().num_elements() != dst->shape().num_elements())
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "CastFloat32ToUint8: shape mismatch");
            }

            const float *in =
                static_cast<const float *>(src.buffer().data());
            std::uint8_t *out =
                static_cast<std::uint8_t *>(dst->buffer().data());

            std::size_t n =
                static_cast<std::size_t>(src.shape().num_elements());
            for (std::size_t i = 0; i < n; ++i)
            {
                // No clamping here, assumes values already in [0,255]
                out[i] = static_cast<std::uint8_t>(in[i]);
            }

            return core::Status::Ok();
        }
} // namespace ptk::operators

