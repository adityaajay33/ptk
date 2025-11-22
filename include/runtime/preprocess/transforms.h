#ifndef RUNTIME_PREPROCESS_TRANSFORMS_H_
#define RUNTIME_PREPROCESS_TRANSFORMS_H_

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/frame.h"
#include "runtime/data/tensor.h"

namespace runtime
{
    namespace preprocess
    {

        Status HwcToChw(const TensorView &src, TensorView *dst);
        Status ChwToHwc(const TensorView &src, TensorView *dst);

        Status RgbToBgr(TensorView *tensor);
        Status BgrToRgb(TensorView *tensor);
        Status RgbToGray(const TensorView &src, TensorView *dst);

        Status CastUint8ToFloat32(const TensorView &src, TensorView *dst);
        Status CastFloat32ToUint8(const TensorView &src, TensorView *dst);

        struct NormalizationParams
        {
            float mean[4];
            float std[4];
            int num_channels;
        };

        Status Normalize(TensorView *tensor, const NormalizationParams &params, TensorLayout layout);

        Status PadToSize(const TensorView &src, int target_h, int target_w, TensorView *dst);

        Status CenterCrop(const TensorView &src, int crop_h, int crop_w, TensorView *dst);

        Status AddBatchDim(const TensorView &src, TensorView *dst);
    };
} // namespace runtime

#endif // RUNTIME_PREPROCESS_TRANSFORMS_H_