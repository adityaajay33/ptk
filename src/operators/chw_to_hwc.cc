#include "operators/chw_to_hwc.h"

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        Status ChwToHwc(const TensorView &src, TensorView *dst)
        {
            if (dst == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "ChwToHwc: dst is null");
            }
            if (src.dtype() != DataType::kFloat32 ||
                dst->dtype() != DataType::kFloat32)
            {
                return Status(StatusCode::kInvalidArgument,
                              "ChwToHwc: expects float32 src and dst");
            }

            const TensorShape &sshape = src.shape();
            const TensorShape &dshape = dst->shape();

            if (sshape.rank() != 3 || dshape.rank() != 3)
            {
                return Status(StatusCode::kInvalidArgument,
                              "ChwToHwc: expects rank 3 tensors");
            }

            const std::int64_t C = sshape.dim(0);
            const std::int64_t H = sshape.dim(1);
            const std::int64_t W = sshape.dim(2);

            if (dshape.dim(0) != H ||
                dshape.dim(1) != W ||
                dshape.dim(2) != C)
            {
                return Status(StatusCode::kInvalidArgument,
                              "ChwToHwc: dst shape must be [H,W,C]");
            }

            const float *src_data =
                static_cast<const float *>(src.buffer().data());
            float *dst_data =
                static_cast<float *>(dst->buffer().data());

            if (src_data == nullptr || dst_data == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "ChwToHwc: null buffer data");
            }

            // CHW to HWC
            // src: (c, h, w) -> (c * H + h) * W + w
            // dst: (h, w, c) -> (h * W + w) * C + c
            for (std::int64_t h = 0; h < H; ++h)
            {
                for (std::int64_t w = 0; w < W; ++w)
                {
                    for (std::int64_t c = 0; c < C; ++c)
                    {
                        const std::size_t src_idx =
                            static_cast<std::size_t>((c * H + h) * W + w);
                        const std::size_t dst_idx =
                            static_cast<std::size_t>((h * W + w) * C + c);
                        dst_data[dst_idx] = src_data[src_idx];
                    }
                }
            }

            return Status::Ok();
        }
    }
}

