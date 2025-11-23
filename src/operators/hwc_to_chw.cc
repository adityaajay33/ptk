#include "operators/hwc_to_chw.h"

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status HwcToChw(const data::TensorView &src, data::TensorView *dst)
        {
            if (dst == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "HwcToChw: dst is null");
            }
            if (src.dtype() != core::DataType::kFloat32 ||
                dst->dtype() != core::DataType::kFloat32)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "HwcToChw: expects float32 src and dst");
            }

            const data::TensorShape &sshape = src.shape();
            const data::TensorShape &dshape = dst->shape();

            if (sshape.rank() != 3 || dshape.rank() != 3)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "HwcToChw: expects rank 3 tensors");
            }

            const std::int64_t H = sshape.dim(0);
            const std::int64_t W = sshape.dim(1);
            const std::int64_t C = sshape.dim(2);

            if (dshape.dim(0) != C ||
                dshape.dim(1) != H ||
                dshape.dim(2) != W)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "HwcToChw: dst shape must be [C,H,W]");
            }

            const float *src_data =
                static_cast<const float *>(src.buffer().data());
            float *dst_data =
                static_cast<float *>(dst->buffer().data());

            if (src_data == nullptr || dst_data == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "HwcToChw: null buffer data");
            }

            // HWC to CHW
            // src: (h, w, c) -> (h * W + w) * C + c
            // dst: (c, h, w) -> (c * H + h) * W + w
            for (std::int64_t c = 0; c < C; ++c)
            {
                for (std::int64_t h = 0; h < H; ++h)
                {
                    for (std::int64_t w = 0; w < W; ++w)
                    {
                        const std::size_t src_idx =
                            static_cast<std::size_t>((h * W + w) * C + c);
                        const std::size_t dst_idx =
                            static_cast<std::size_t>((c * H + h) * W + w);
                        dst_data[dst_idx] = src_data[src_idx];
                    }
                }
            }

            return core::Status::Ok();
        }
    }
}

