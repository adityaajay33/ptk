#include "operators/normalize.h"

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk
{
    namespace operators
    {
        core::Status Normalize(data::TensorView *tensor,
                         const NormalizationParams &params,
                         core::TensorLayout layout)
        {
            if (tensor == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "Normalize: tensor is null");
            }

            if (tensor->dtype() != core::DataType::kFloat32)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "Normalize: expects float32 tensor");
            }

            const data::TensorShape &shape = tensor->shape();
            const std::size_t rank = shape.rank();

            std::int64_t N = 1;
            std::int64_t C = 0;
            std::int64_t H = 0;
            std::int64_t W = 0;

            switch (layout)
            {
            case core::TensorLayout::kHwc:
            {
                if (rank != 3)
                {
                    return core::Status(core::StatusCode::kInvalidArgument,
                                  "Normalize: HWC layout expects rank 3 tensor");
                }
                H = shape.dim(0);
                W = shape.dim(1);
                C = shape.dim(2);
                N = 1;
                break;
            }
            case core::TensorLayout::kChw:
            {
                if (rank != 3)
                {
                    return core::Status(core::StatusCode::kInvalidArgument,
                                  "Normalize: CHW layout expects rank 3 tensor");
                }
                C = shape.dim(0);
                H = shape.dim(1);
                W = shape.dim(2);
                N = 1;
                break;
            }
            case core::TensorLayout::kNhwc:
            {
                if (rank != 4)
                {
                    return core::Status(core::StatusCode::kInvalidArgument,
                                  "Normalize: NHWC layout expects rank 4 tensor");
                }
                N = shape.dim(0);
                H = shape.dim(1);
                W = shape.dim(2);
                C = shape.dim(3);
                break;
            }
            case core::TensorLayout::kNchw:
            {
                if (rank != 4)
                {
                    return core::Status(core::StatusCode::kInvalidArgument,
                                  "Normalize: NCHW layout expects rank 4 tensor");
                }
                N = shape.dim(0);
                C = shape.dim(1);
                H = shape.dim(2);
                W = shape.dim(3);
                break;
            }
            default:
                return core::Status(core::StatusCode::kInvalidArgument,
                              "Normalize: unsupported TensorLayout");
            }

            if (C <= 0 || H <= 0 || W <= 0 || N <= 0)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "Normalize: non positive dimension");
            }

            if (params.num_channels <= 0 ||
                params.num_channels != C)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "Normalize: num_channels must match tensor channels");
            }

            for (int c = 0; c < params.num_channels; ++c)
            {
                if (params.std[c] == 0.0f)
                {
                    return core::Status(core::StatusCode::kInvalidArgument,
                                  "Normalize: std for a channel is zero");
                }
            }

            float *data =
                static_cast<float *>(tensor->buffer().data());
            if (data == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                              "Normalize: tensor buffer data is null");
            }

            switch (layout)
            {
            case core::TensorLayout::kHwc:
            {
                // (h, w, c) -> (h * W + w) * C + c
                for (std::int64_t h = 0; h < H; ++h)
                {
                    for (std::int64_t w = 0; w < W; ++w)
                    {
                        for (std::int64_t c = 0; c < C; ++c)
                        {
                            const std::size_t idx =
                                static_cast<std::size_t>((h * W + w) * C + c);
                            const float m = params.mean[c];
                            const float s = params.std[c];
                            data[idx] = (data[idx] - m) / s;
                        }
                    }
                }
                break;
            }

            case core::TensorLayout::kChw:
            {
                // (c, h, w) -> (c * H + h) * W + w
                for (std::int64_t c = 0; c < C; ++c)
                {
                    for (std::int64_t h = 0; h < H; ++h)
                    {
                        for (std::int64_t w = 0; w < W; ++w)
                        {
                            const std::size_t idx =
                                static_cast<std::size_t>((c * H + h) * W + w);
                            const float m = params.mean[c];
                            const float s = params.std[c];
                            data[idx] = (data[idx] - m) / s;
                        }
                    }
                }
                break;
            }

            case core::TensorLayout::kNhwc:
            {
                // (n, h, w, c) -> (((n * H + h) * W + w) * C + c)
                for (std::int64_t n = 0; n < N; ++n)
                {
                    for (std::int64_t h = 0; h < H; ++h)
                    {
                        for (std::int64_t w = 0; w < W; ++w)
                        {
                            for (std::int64_t c = 0; c < C; ++c)
                            {
                                const std::size_t idx =
                                    static_cast<std::size_t>(
                                        (((n * H + h) * W + w) * C + c));
                                const float m = params.mean[c];
                                const float s = params.std[c];
                                data[idx] = (data[idx] - m) / s;
                            }
                        }
                    }
                }
                break;
            }

            case core::TensorLayout::kNchw:
            {
                // (n, c, h, w) -> (((n * C + c) * H + h) * W + w)
                for (std::int64_t n = 0; n < N; ++n)
                {
                    for (std::int64_t c = 0; c < C; ++c)
                    {
                        for (std::int64_t h = 0; h < H; ++h)
                        {
                            for (std::int64_t w = 0; w < W; ++w)
                            {
                                const std::size_t idx =
                                    static_cast<std::size_t>(
                                        (((n * C + c) * H + h) * W + w));
                                const float m = params.mean[c];
                                const float s = params.std[c];
                                data[idx] = (data[idx] - m) / s;
                            }
                        }
                    }
                }
                break;
            }

            default:
                return core::Status(core::StatusCode::kInternal,
                              "Normalize: unreachable layout branch");
            }

            return core::Status::Ok();
        }
    }
}

