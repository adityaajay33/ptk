#ifndef RUNTIME_CORE_RUNTIME_CONTEXT_H_
#define RUNTIME_CORE_RUNTIME_CONTEXT_H_

#include <string_view>

#include "runtime/core/status.h"

namespace ptk {
namespace core {

    enum class LogSeverity {

        kInfo = 0,
        kWarning,
        kError,
    };


    struct RuntimeContextOptions {
        //output stream for logging
        std::FILE* info_stream = nullptr;
        std::FILE* error_stream = nullptr;
    };

    class RuntimeContext {

        public:

            RuntimeContext();
            ~RuntimeContext();

            RuntimeContext(const RuntimeContext&) = delete;
            RuntimeContext& operator=(const RuntimeContext&) = delete;
            RuntimeContext(RuntimeContext&&) = delete;
            RuntimeContext& operator=(RuntimeContext&&) = delete;

            //call before use
            Status Init(const RuntimeContextOptions& options);

            //clean up resources
            void Shutdown();

            // clock
            std::int64_t NowNanoseconds() const;

            //api for logging
            void Log(LogSeverity severity, std::string_view message) const;
            void LogInfo(std::string_view message) const { Log(LogSeverity::kInfo, message); }
            void LogWarning(std::string_view message) const { Log(LogSeverity::kWarning, message); }
            void LogError(std::string_view message) const { Log(LogSeverity::kError, message); }

            bool initialized() const { return initialized_; }

        private:
            bool initialized_;
            RuntimeContextOptions options_;
    };

} // namespace core
}


#endif //RUNTIME_CORE_RUNTIME_CONTEXT_H_