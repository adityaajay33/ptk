#ifndef OPERATORS_NORMALIZATION_PARAMS_H_
#define OPERATORS_NORMALIZATION_PARAMS_H_

namespace ptk
{
    namespace operators
    {
        struct NormalizationParams
        {
            float mean[4];
            float std[4];
            int num_channels;
        };
    }
}

#endif // OPERATORS_NORMALIZATION_PARAMS_H_

