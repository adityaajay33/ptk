#include "runtime/preprocess/transforms.h"
#include "runtime/preprocess/preprocessor.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/status.h"

namespace runtime {

Preprocessor::Preprocessor(const PreprocessorConfig& config)
    : context_(nullptr),
      input_(nullptr),
      output_(nullptr),
      config_(config),
      float_buffer_(),
      uint8_temp_(),
      output_frame_() {}

Status Preprocessor::Init(RuntimeContext* context) {
  if (context == nullptr) {
    return Status(StatusCode::kInvalidArgument, "Context is null");
  }
  context_ = context;
  return Status::Ok();
}

Status Preprocessor::Start() {
  if (input_ == nullptr || !input_->is_bound() ||
      output_ == nullptr || !output_->is_bound()) {
    return Status(
        StatusCode::kFailedPrecondition,
        "Preprocessor ports not bound");
  }
  return Status::Ok();
}

void Preprocessor::Tick() {
  if (context_ == nullptr) {
    return;
  }
  if (input_ == nullptr || !input_->is_bound()) {
    context_->LogError("Preprocessor: input port not bound");
    return;
  }
  if (output_ == nullptr || !output_->is_bound()) {
    context_->LogError("Preprocessor: output port not bound");
    return;
  }

  const Frame* in = input_->get();
  Frame* out = output_->get();

  if (in == nullptr || out == nullptr) {
    context_->LogError("Preprocessor: null frame from port");
    return;
  }

  // Copy basic metadata.
  out->frame_index = in->frame_index;
  out->timestamp_ns = in->timestamp_ns;
  out->camera_id = in->camera_id;
  out->pixel_format = in->pixel_format;

  // Your Frame::image is a plain TensorView, not optional.
  // Use TensorView::empty() to check validity.
  if (in->image.empty()) {
    context_->LogError("Preprocessor: input frame has empty image tensor");
    return;
  }
  if (out->image.empty()) {
    context_->LogError("Preprocessor: output frame has empty image tensor");
    return;
  }

  const TensorView& src = in->image;
  TensorView& dst = out->image;

  // For now: simple uint8 -> float32 cast.
  Status s = preprocess::CastUint8ToFloat32(src, &dst);
  if (!s.ok()) {
    context_->LogError("Preprocessor: CastUint8ToFloat32 failed");
    return;
  }

}

}  // namespace runtime