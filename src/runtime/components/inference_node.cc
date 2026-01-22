#include "runtime/components/inference_node.h"
#include "runtime/core/logger.h"
#include "engines/onnx_engine.h"
#ifdef PTK_ENABLE_CUDA
#include "engines/trt_engine.h"
#endif
#include "tasks/detection_contract.h"
#include "tasks/segmentation_contract.h"
#include <rclcpp_components/register_node_macro.hpp>
#include <chrono>

namespace ptk::components
{

    InferenceNode::InferenceNode(const rclcpp::NodeOptions &options)
        : ComponentInterface("inference_node", options),
          context_(nullptr),
          input_(nullptr),
          output_(nullptr),
          model_path_(""),
          task_type_("detection"),
          confidence_threshold_(0.5f),
          nms_threshold_(0.45f),
          max_detections_(100),
          total_inferences_(0),
          total_inference_time_ms_(0.0)
    {
        LoadParametersFromROS();
    }

    void InferenceNode::LoadParametersFromROS()
    {
        // Model configuration
        this->declare_parameter("model_path", "");
        this->declare_parameter("task_type", "detection");

        // Engine configuration
        this->declare_parameter("backend", "onnx");
        this->declare_parameter("device_id", 0);
        this->declare_parameter("execution_provider", "cpu");
        this->declare_parameter("enable_dynamic_shapes", false);
        this->declare_parameter("verbose", false);

        // Task parameters
        this->declare_parameter("confidence_threshold", 0.5);
        this->declare_parameter("nms_threshold", 0.45);
        this->declare_parameter("max_detections", 100);

        // Get parameters
        model_path_ = this->get_parameter("model_path").as_string();
        task_type_ = this->get_parameter("task_type").as_string();

        // Engine config
        std::string backend = this->get_parameter("backend").as_string();
        if (backend == "onnx")
        {
            engine_config_.backend = perception::EngineBackend::OnnxRuntime;
        }
        else if (backend == "tensorrt")
        {
            engine_config_.backend = perception::EngineBackend::TensorRTNative;
        }

        std::string exec_provider = this->get_parameter("execution_provider").as_string();
        if (exec_provider == "cpu")
        {
            engine_config_.onnx_execution_provider = perception::OnnxRuntimeExecutionProvider::Cpu;
        }
        else if (exec_provider == "cuda")
        {
            engine_config_.onnx_execution_provider = perception::OnnxRuntimeExecutionProvider::Cuda;
        }
        else if (exec_provider == "tensorrt")
        {
            engine_config_.onnx_execution_provider = perception::OnnxRuntimeExecutionProvider::TensorRTEP;
        }

        engine_config_.device_id = this->get_parameter("device_id").as_int();
        engine_config_.enable_dynamic_shapes = this->get_parameter("enable_dynamic_shapes").as_bool();
        engine_config_.verbose = this->get_parameter("verbose").as_bool();

        // Task parameters
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        nms_threshold_ = this->get_parameter("nms_threshold").as_double();
        max_detections_ = this->get_parameter("max_detections").as_int();
    }

    void InferenceNode::BindInput(core::InputPort<data::Frame> *in)
    {
        input_ = in;
    }

    void InferenceNode::BindOutput(core::OutputPort<tasks::TaskOutput> *out)
    {
        output_ = out;
    }

    void InferenceNode::SetEngine(std::unique_ptr<perception::Engine> engine)
    {
        engine_ = std::move(engine);
    }

    void InferenceNode::SetTaskContract(std::unique_ptr<tasks::TaskContract> contract)
    {
        task_contract_ = std::move(contract);
    }

    core::Status InferenceNode::ValidateConfiguration()
    {
        if (model_path_.empty())
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "Model path not specified. Set 'model_path' parameter.");
        }

        if (input_ == nullptr)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "Input port not bound. Call BindInput() before Init().");
        }

        if (output_ == nullptr)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "Output port not bound. Call BindOutput() before Init().");
        }

        return core::Status::Ok();
    }

    core::Status InferenceNode::CreateEngine()
    {
        if (engine_)
        {
            return core::Status::Ok();
        }

        if (engine_config_.backend == perception::EngineBackend::OnnxRuntime)
        {
            engine_ = std::make_unique<perception::OnnxEngine>(engine_config_);
        }
#ifdef PTK_ENABLE_CUDA
        else if (engine_config_.backend == perception::EngineBackend::TensorRTNative)
        {
            engine_ = std::make_unique<perception::TrtEngine>(engine_config_);
        }
#endif
        else
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "Unsupported backend");
        }

        return core::Status::Ok();
    }

    core::Status InferenceNode::CreateTaskContract()
    {
        if (task_contract_)
        {
            return core::Status::Ok();
        }

        if (task_type_ == "detection")
        {
            // TODO: Load class labels from parameter
            std::vector<std::string> class_labels = {"person", "car", "dog", "cat"};
            task_contract_ = std::make_unique<tasks::ObjectDetectionContract>(
                class_labels,
                tasks::CoordinateSystem::kImagePixels);
        }
        else
        {
            // segmentation currenlty invalid
            return core::Status(core::StatusCode::kInvalidArgument,
                                "Unsupported task type: " + task_type_);
        }

        return core::Status::Ok();
    }

    core::Status InferenceNode::LoadModel()
    {
        if (!engine_)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "Engine not created. Call CreateEngine() first.");
        }

        auto engine_status = engine_->Load(model_path_);
        if (!engine_status.ok())
        {
            RCLCPP_ERROR(this->get_logger(), "Model load failed: %s", engine_status.message().c_str());
            return core::Status(engine_status.code(), engine_status.message());
        }

        return core::Status::Ok();
    }

    core::Status InferenceNode::Init(core::RuntimeContext *context)
    {
        context_ = context;

        auto status = ValidateConfiguration();
        if (!status.ok())
        {
            return status;
        }

        if (!engine_)
        {
            status = CreateEngine();
            if (!status.ok())
            {
                return status;
            }
        }

        if (!task_contract_)
        {
            status = CreateTaskContract();
            if (!status.ok())
            {
                return status;
            }
        }

        status = LoadModel();
        if (!status.ok())
        {
            return status;
        }

        status = task_contract_->ValidateModel(*engine_);
        if (!status.ok())
        {
            RCLCPP_WARN(this->get_logger(), "Model validation: %s", status.message().c_str());
        }

        RCLCPP_INFO(this->get_logger(), "Initialized [model=%s, task=%s]",
                    model_path_.c_str(), task_type_.c_str());
        return core::Status::Ok();
    }

    core::Status InferenceNode::Start()
    {
        total_inferences_ = 0;
        total_inference_time_ms_ = 0.0;
        return core::Status::Ok();
    }

    core::Status InferenceNode::Stop()
    {
        PublishStatistics();
        return core::Status::Ok();
    }

    void InferenceNode::Tick()
    {
        // Read input frame from port
        if (!input_ || !input_->is_bound())
        {
            return;
        }

        const data::Frame *frame_ptr = input_->get();
        if (!frame_ptr)
        {
            return;
        }

        const data::Frame &input_frame = *frame_ptr;

        // Prepare task input
        tasks::TaskInput task_input;
        task_input.frame = input_frame;
        task_input.params.confidence_threshold = confidence_threshold_;
        task_input.params.nms_threshold = nms_threshold_;
        task_input.params.max_detections = max_detections_;

        auto status = task_contract_->ValidateInput(task_input);
        if (!status.ok())
        {
            RCLCPP_ERROR(this->get_logger(), "Invalid input: %s", status.message().c_str());
            return;
        }

        std::vector<data::TensorView> raw_inputs;
        status = task_contract_->PreProcess(task_input, &raw_inputs);
        if (!status.ok())
        {
            RCLCPP_ERROR(this->get_logger(), "Preprocess failed: %s", status.message().c_str());
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<data::TensorView> raw_outputs;
        auto engine_status = engine_->Infer(raw_inputs, raw_outputs);
        if (!engine_status.ok())
        {
            RCLCPP_ERROR(this->get_logger(), "Inference failed: %s", engine_status.message().c_str());
            return;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        float inference_time_ms = std::chrono::duration<float, std::milli>(
                                      end_time - start_time)
                                      .count();

        output_result_ = tasks::TaskOutput();
        output_result_.timestamp_ns = input_frame.timestamp_ns;
        output_result_.frame_index = input_frame.frame_index;
        output_result_.inference_time_ms = inference_time_ms;
        output_result_.original_width = input_frame.image.shape().dims()[1];
        output_result_.original_height = input_frame.image.shape().dims()[0];

        status = task_contract_->PostProcess(raw_outputs, task_input, &output_result_);
        if (!status.ok())
        {
            RCLCPP_ERROR(this->get_logger(), "Postprocess failed: %s", status.message().c_str());
            return;
        }

        status = task_contract_->ValidateOutput(output_result_);
        if (!status.ok())
        {
            RCLCPP_WARN(this->get_logger(), "Output validation: %s", status.message().c_str());
        }

        output_result_.success = true;
        output_result_.task_type = task_type_;

        total_inferences_++;
        total_inference_time_ms_ += inference_time_ms;

        if (output_ && output_->is_bound())
        {
            output_->Bind(&output_result_);
        }
    }

    void InferenceNode::PublishStatistics()
    {
        if (total_inferences_ > 0)
        {
            float avg_time = total_inference_time_ms_ / total_inferences_;
            RCLCPP_INFO(this->get_logger(), "Statistics: %ld frames, %.2f ms avg, %.1f fps",
                        total_inferences_, avg_time, 1000.0f / avg_time);
        }
    }

} // namespace ptk::components

RCLCPP_COMPONENTS_REGISTER_NODE(ptk::components::InferenceNode)
