#include "runtime/preprocess/transforms.h"

#include <cstddef>
#include <cstdint>

namespace runtime
{
    namespace preprocess
    {

        Status HwcToChw(const TensorView &src, TensorView *dst)
        {
            if (dst == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "HwcToChw: dst is null");
            }
            if (src.dtype() != DataType::kFloat32 ||
                dst->dtype() != DataType::kFloat32)
            {
                return Status(StatusCode::kInvalidArgument,
                              "HwcToChw: expects float32 src and dst");
            }

            const TensorShape &sshape = src.shape();
            const TensorShape &dshape = dst->shape();

            if (sshape.rank() != 3 || dshape.rank() != 3)
            {
                return Status(StatusCode::kInvalidArgument,
                              "HwcToChw: expects rank 3 tensors");
            }

            const std::int64_t H = sshape.dim(0);
            const std::int64_t W = sshape.dim(1);
            const std::int64_t C = sshape.dim(2);

            if (dshape.dim(0) != C ||
                dshape.dim(1) != H ||
                dshape.dim(2) != W)
            {
                return Status(StatusCode::kInvalidArgument,
                              "HwcToChw: dst shape must be [C,H,W]");
            }

            const float *src_data =
                static_cast<const float *>(src.buffer().data());
            float *dst_data =
                static_cast<float *>(dst->buffer().data());

            if (src_data == nullptr || dst_data == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
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

            return Status::Ok();
        }

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

        Status CastUint8ToFloat32(const TensorView &src, TensorView *dst)
        {
            if (dst == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "CastUint8ToFloat32: dst is null");
            }
            if (src.dtype() != DataType::kUint8 ||
                dst->dtype() != DataType::kFloat32)
            {
                return Status(StatusCode::kInvalidArgument,
                              "CastUint8ToFloat32: invalid data types");
            }

            if (src.shape().num_elements() != dst->shape().num_elements())
            {
                return Status(StatusCode::kInvalidArgument,
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

            return Status::Ok();
        }

        Status Normalize(TensorView *tensor,
                         const NormalizationParams &params,
                         TensorLayout layout)
        {
            if (tensor == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "Normalize: tensor is null");
            }

            if (tensor->dtype() != DataType::kFloat32)
            {
                return Status(StatusCode::kInvalidArgument,
                              "Normalize: expects float32 tensor");
            }

            const TensorShape &shape = tensor->shape();
            const std::size_t rank = shape.rank();

            std::int64_t N = 1;
            std::int64_t C = 0;
            std::int64_t H = 0;
            std::int64_t W = 0;

            switch (layout)
            {
            case TensorLayout::kHwc:
            {
                if (rank != 3)
                {
                    return Status(StatusCode::kInvalidArgument,
                                  "Normalize: HWC layout expects rank 3 tensor");
                }
                H = shape.dim(0);
                W = shape.dim(1);
                C = shape.dim(2);
                N = 1;
                break;
            }
            case TensorLayout::kChw:
            {
                if (rank != 3)
                {
                    return Status(StatusCode::kInvalidArgument,
                                  "Normalize: CHW layout expects rank 3 tensor");
                }
                C = shape.dim(0);
                H = shape.dim(1);
                W = shape.dim(2);
                N = 1;
                break;
            }
            case TensorLayout::kNhwc:
            {
                if (rank != 4)
                {
                    return Status(StatusCode::kInvalidArgument,
                                  "Normalize: NHWC layout expects rank 4 tensor");
                }
                N = shape.dim(0);
                H = shape.dim(1);
                W = shape.dim(2);
                C = shape.dim(3);
                break;
            }
            case TensorLayout::kNchw:
            {
                if (rank != 4)
                {
                    return Status(StatusCode::kInvalidArgument,
                                  "Normalize: NCHW layout expects rank 4 tensor");
                }
                N = shape.dim(0);
                C = shape.dim(1);
                H = shape.dim(2);
                W = shape.dim(3);
                break;
            }
            default:
                return Status(StatusCode::kInvalidArgument,
                              "Normalize: unsupported TensorLayout");
            }

            if (C <= 0 || H <= 0 || W <= 0 || N <= 0)
            {
                return Status(StatusCode::kInvalidArgument,
                              "Normalize: non positive dimension");
            }

            if (params.num_channels <= 0 ||
                params.num_channels != C)
            {
                return Status(StatusCode::kInvalidArgument,
                              "Normalize: num_channels must match tensor channels");
            }

            for (int c = 0; c < params.num_channels; ++c)
            {
                if (params.std[c] == 0.0f)
                {
                    return Status(StatusCode::kInvalidArgument,
                                  "Normalize: std for a channel is zero");
                }
            }

            float *data =
                static_cast<float *>(tensor->buffer().data());
            if (data == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "Normalize: tensor buffer data is null");
            }

            switch (layout)
            {
            case TensorLayout::kHwc:
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

            case TensorLayout::kChw:
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

            case TensorLayout::kNhwc:
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

            case TensorLayout::kNchw:
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
                return Status(StatusCode::kInternal,
                              "Normalize: unreachable layout branch");
            }

            return Status::Ok();
        }

        Status CastFloat32ToUint8(const TensorView &src, TensorView *dst)
        {
            if (dst == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "CastFloat32ToUint8: dst is null");
            }
            if (src.dtype() != DataType::kFloat32 ||
                dst->dtype() != DataType::kUint8)
            {
                return Status(StatusCode::kInvalidArgument,
                              "CastFloat32ToUint8: invalid data types");
            }

            if (src.shape().num_elements() != dst->shape().num_elements())
            {
                return Status(StatusCode::kInvalidArgument,
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

            return Status::Ok();
        }

        Status RgbToGray(const TensorView &src, TensorView *dst)
        {
            if (dst == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToGray: dst is null");
            }

            if (src.dtype() != DataType::kFloat32 ||
                dst->dtype() != DataType::kFloat32)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToGray: expects float32 src and dst");
            }

            const TensorShape &sshape = src.shape();
            const TensorShape &dshape = dst->shape();

            if (sshape.rank() != 3 || dshape.rank() != 3)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToGray: expects rank 3 HWC tensors");
            }

            const std::int64_t H = sshape.dim(0);
            const std::int64_t W = sshape.dim(1);
            const std::int64_t C = sshape.dim(2);

            if (C != 3)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToGray: src must have 3 channels");
            }

            if (dshape.dim(0) != H ||
                dshape.dim(1) != W ||
                dshape.dim(2) != 1)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToGray: dst shape must be [H,W,1]");
            }

            const float *src_data =
                static_cast<const float *>(src.buffer().data());
            float *dst_data =
                static_cast<float *>(dst->buffer().data());

            if (src_data == nullptr || dst_data == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
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

            return Status::Ok();
        }

        Status RgbToBgr(TensorView *tensor)
        {
            if (tensor == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToBgr: tensor is null");
            }

            if (tensor->dtype() != DataType::kFloat32)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToBgr: expects float32 tensor");
            }

            const TensorShape &shape = tensor->shape();
            if (shape.rank() != 3)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToBgr: expects rank 3 HWC tensor");
            }

            const std::int64_t H = shape.dim(0);
            const std::int64_t W = shape.dim(1);
            const std::int64_t C = shape.dim(2);

            if (C != 3)
            {
                return Status(StatusCode::kInvalidArgument,
                              "RgbToBgr: expects 3 channel tensor");
            }

            float *data =
                static_cast<float *>(tensor->buffer().data());
            if (data == nullptr)
            {
                return Status(StatusCode::kInvalidArgument,
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

            return Status::Ok();
        }

        Status BgrToRgb(TensorView *tensor)
        {
            // Same swap as RgbToBgr
            return RgbToBgr(tensor);
        }

    } // namespace preprocess
} // namespace runtime