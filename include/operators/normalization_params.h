#pragma once

namespace ptk::operators
{
        struct NormalizationParams
        {
            float mean[4];
            float std[4];
            int num_channels;
        };
}