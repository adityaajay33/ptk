#include "operators/rgb_to_bgr.h"

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status RgbToBgr(data::TensorView *tensor)
        {
            if (tensor == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToBgr: tensor is null");
            }

            if (tensor->dtype() != core::DataType::kFloat32)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToBgr: expects float32 tensor");
            }

            const data::TensorShape &shape = tensor->shape();
            if (shape.rank() != 3)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToBgr: expects rank 3 HWC tensor");
            }

            const std::int64_t H = shape.dim(0);
            const std::int64_t W = shape.dim(1);
            const std::int64_t C = shape.dim(2);

            if (C != 3)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToBgr: expects 3 channel tensor");
            }

            float *data =
                static_cast<float *>(tensor->buffer().data());
            if (data == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "RgbToBgr: tensor buffer data is null");
            }

            for (std::int64_t h = 0; h < H; ++h)
            {
                for (std::int64_t w = 0; w < W; ++w)
                {
                    const std::size_t base =
                        static_cast<std::size_t>((h * W + w) * 3);
                    float r = data[base + 0];
                    float g = data[base + 1];
                    float b = data[base + 2];
                    data[base + 0] = b;
                    data[base + 1] = g;
                    data[base + 2] = r;
                }
            }

            return core::Status::Ok();
        }
    }
}

