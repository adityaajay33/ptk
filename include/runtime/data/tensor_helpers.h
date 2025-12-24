#pragma once

#include "tensor.h"

namespace ptk::data {

    inline std::size_t TensorByteSize(const TensorView& t){
        return t.num_elements() * t.element_size();
    }
}
