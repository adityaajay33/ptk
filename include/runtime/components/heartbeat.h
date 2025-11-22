#ifndef RUNTIME_COMPONENTS_HEARTBEAT_H_
#define RUNTIME_COMPONENTS_HEARTBEAT_H_

#include "runtime/components/component_interface.h"

namespace ptk {

class Heartbeat : public ComponentInterface {
 public:
  Heartbeat();
  ~Heartbeat() override = default;

  Status Init(RuntimeContext* context) override;
  Status Start() override;
  void Stop() override;
  void Tick() override;

 private:
  RuntimeContext* context_;
  int count_;
};

}  // namespace ptk

#endif  // RUNTIME_COMPONENTS_HEARTBEAT_H_