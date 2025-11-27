#include "operators/rgb_to_gray.h"

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk::operators
{
        core::Status RgbToGray(const data::TensorView &src, data::TensorView *dst)
        {
            if (dst == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToGray: dst is null");
            }

            if (src.dtype() != core::DataType::kFloat32 ||
                dst->dtype() != core::DataType::kFloat32)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToGray: expects float32 src and dst");
            }

            const data::TensorShape &sshape = src.shape();
            const data::TensorShape &dshape = dst->shape();

            if (sshape.rank() != 3 || dshape.rank() != 3)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToGray: expects rank 3 HWC tensors");
            }

            const std::int64_t H = sshape.dim(0);
            const std::int64_t W = sshape.dim(1);
            const std::int64_t C = sshape.dim(2);

            if (C != 3)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToGray: src must have 3 channels");
            }

            if (dshape.dim(0) != H ||
                dshape.dim(1) != W ||
                dshape.dim(2) != 1)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToGray: dst shape must be [H,W,1]");
            }

            const float *src_data =
                static_cast<const float *>(src.buffer().data());
            float *dst_data =
                static_cast<float *>(dst->buffer().data());

            if (src_data == nullptr || dst_data == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToGray: null buffer data");
            }

            const float kR = 0.299f;
            const float kG = 0.587f;
            const float kB = 0.114f;

            for (std::int64_t h = 0; h < H; ++h)
            {
                for (std::int64_t w = 0; w < W; ++w)
                {
                    const std::size_t src_base =
                        static_cast<std::size_t>((h * W + w) * 3);
                    const std::size_t dst_idx =
                        static_cast<std::size_t>((h * W + w));

                    const float r = src_data[src_base + 0];
                    const float g = src_data[src_base + 1];
                    const float b = src_data[src_base + 2];

                    dst_data[dst_idx] = kR * r + kG * g + kB * b;
                }
            }

            return core::Status::Ok();
        }
} // namespace ptk::operators

