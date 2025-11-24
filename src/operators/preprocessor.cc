#include "operators/preprocessor.h"
#include "operators/hwc_to_chw.h"
#include "operators/chw_to_hwc.h"
#include "operators/cast_uint8_to_float32.h"
#include "operators/cast_float32_to_uint8.h"
#include "operators/normalize.h"
#include "operators/rgb_to_gray.h"
#include "operators/rgb_to_bgr.h"
#include "operators/bgr_to_rgb.h"
#include "operators/pad_to_size.h"
#include "operators/center_crop.h"
#include "operators/add_batch_dim.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/status.h"

namespace ptk {

Preprocessor::Preprocessor(const PreprocessorConfig& config)
    : context_(nullptr),
      input_(nullptr),
      output_(nullptr),
      config_(config),
      float_buffer_(),
      uint8_temp_(),
      output_frame_() {}

core::Status Preprocessor::Init(core::RuntimeContext* context) {
  if (context == nullptr) {
    return core::Status(core::StatusCode::kInvalidArgument, "Context is null");
  }
  context_ = context;
  return core::Status::Ok();
}

core::Status Preprocessor::Start() {
  if (input_ == nullptr || !input_->is_bound() ||
      output_ == nullptr || !output_->is_bound()) {
    return core::Status(
        core::StatusCode::kFailedPrecondition,
        "Preprocessor ports not bound");
  }
  return core::Status::Ok();
}

core::Status Preprocessor::Stop() {
  return core::Status::Ok();
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

  const data::Frame* in = input_->get();
  data::Frame* out = output_->get();

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

  const data::TensorView& src = in->image;
  data::TensorView& dst = out->image;

  // For now: simple uint8 -> float32 cast.
  core::Status s = operators::CastUint8ToFloat32(src, &dst);
  if (!s.ok()) {
    context_->LogError("Preprocessor: CastUint8ToFloat32 failed");
    return;
  }

}

}  // namespace ptk

