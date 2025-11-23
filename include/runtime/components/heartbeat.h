#ifndef RUNTIME_COMPONENTS_HEARTBEAT_H_
#define RUNTIME_COMPONENTS_HEARTBEAT_H_

#include "runtime/components/component_interface.h"

namespace ptk {
namespace components {

class Heartbeat : public ComponentInterface {
 public:
  Heartbeat();
  ~Heartbeat() override = default;

  core::Status Init(core::RuntimeContext* context) override;
  core::Status Start() override;
  void Stop() override;
  void Tick() override;

 private:
  core::RuntimeContext* context_;
  int count_;
};

}  // namespace components
}  // namespace ptk

#endif  // RUNTIME_COMPONENTS_HEARTBEAT_H_