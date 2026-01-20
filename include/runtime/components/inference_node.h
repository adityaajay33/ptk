#pragma once

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include "engines/engine.h"
#include "engines/engine_config.h"
#include "tasks/task_contract.h"
#include "tasks/task_output.h"
#include <memory>
#include <string>

namespace ptk::components
{

    class InferenceNode : public ComponentInterface
    {
    public:
        explicit InferenceNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
        ~InferenceNode() override = default;

        void BindInput(core::InputPort<data::Frame> *in);
        void BindOutput(core::OutputPort<tasks::TaskOutput> *out);

        core::Status Init(core::RuntimeContext *context) override;
        core::Status Start() override;
        core::Status Stop() override;
        void Tick() override;

        void SetEngine(std::unique_ptr<perception::Engine> engine);
        void SetTaskContract(std::unique_ptr<tasks::TaskContract> contract);

    private:
        core::RuntimeContext *context_;

        core::InputPort<data::Frame> *input_;
        core::OutputPort<tasks::TaskOutput> *output_;

        std::unique_ptr<perception::Engine> engine_;

        std::unique_ptr<tasks::TaskContract> task_contract_;

        std::string model_path_;
        std::string task_type_;
        perception::EngineConfig engine_config_;

        tasks::TaskOutput output_result_;

        float confidence_threshold_;
        float nms_threshold_;
        int max_detections_;

        int64_t total_inferences_;
        double total_inference_time_ms_;

        core::Status LoadModel();
        core::Status CreateEngine();
        core::Status CreateTaskContract();
        core::Status ValidateConfiguration();
        void LoadParametersFromROS();
        void PublishStatistics();
    };

} // namespace ptk::components
