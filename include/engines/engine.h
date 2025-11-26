#pragma once

#include <string>
#include <vector>
#include <memory>

namespace ptk
{

    namespace perception
    {
        struct EngineTensor {
            void* data;
            std::vector<int64_t> shape;
            int element_type;
        };

        class Engine {
            public:
                virtual ~Engine() = default;

                virtual bool Load(const std::string& model_path) = 0;

                virtual bool Infer(const std::vector<EngineTensor>& inputs,std::vector<EngineTensor>* outputs) = 0;

                virtual std::vector<std::string> InputNames() const = 0;
                virtual std::vector<std::string> OutputNames() const = 0;
        };
    }
}