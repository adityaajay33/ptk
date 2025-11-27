#pragma once
#include <memory>
#include "engine.h"
#include "engine_config.h"

namespace ptk::perception
{

    std::unique_ptr<Engine> CreateEngine(const EngineConfig &config);

}