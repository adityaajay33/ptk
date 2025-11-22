#ifndef OPERATORS_PREPROCESSOR_H_
#define OPERATORS_PREPROCESSOR_H_

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include "runtime/core/types.h"
#include "operators/transforms.h"

namespace ptk {

    struct PreprocessorConfig {
        TensorLayout input_layout;
        PixelFormat input_format;
        DataType input_type;

        TensorLayout output_layout;
        PixelFormat output_format;
        DataType output_type;

        bool convert_rgb_to_bgr;
        bool normalize;
        bool add_batch_dimension;
        bool to_grayscale;
        operators::NormalizationParams norm;

        int target_height;
        int target_width;
    };

    class Preprocessor : public ComponentInterface {
        public:
            explicit Preprocessor(const PreprocessorConfig& config);
            ~Preprocessor() override = default;

            void BindInput(InputPort<Frame>* in);
            void BindOutput(OutputPort<Frame>* out);

            Status Init(RuntimeContext* context) override;
            Status Start() override;
            void Stop() override;
            void Tick() override;

        private:
            RuntimeContext* context_;
            InputPort<Frame>* input_;
            OutputPort<Frame>* output_;
            PreprocessorConfig config_;

            std::vector<float> float_buffer_;
            std::vector<std::uint8_t> uint8_temp_;

            Frame output_frame_;
    };
}

#endif // OPERATORS_PREPROCESSOR_H_

