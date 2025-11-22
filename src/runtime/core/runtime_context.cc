#include "runtime/core/runtime_context.h"

#include <chrono>

namespace runtime {

    namespace{

        const char* Severity(LogSeverity severity){
            switch (severity){
                case LogSeverity::kInfo:
                    return "INFO";
                case LogSeverity::kWarning:
                    return "WARNING";
                case LogSeverity::kError:
                    return "ERROR";
            }
            return "UNKNOWN";
        }
    } 

    RuntimeContext::RuntimeContext() : options_(), initialized_(false) {}

    RuntimeContext::~RuntimeContext() { Shutdown(); }

    Status RuntimeContext::Init(const RuntimeContextOptions& options){

        if (initialized_){
            return Status(StatusCode::kFailedPrecondition, "RuntimeContext::Init() called more than once");
        }

        options_ = options;

        if (options_.info_stream == nullptr){
            options_.info_stream = stdout;
        }
        if  (options_.error_stream == nullptr){
            options_.error_stream = stderr;
        }

        initialized_ = true;

        return Status::Ok();
    }

    void RuntimeContext::Shutdown(){
        if (initialized_){
            return;
        }
    }

    std::int64_t RuntimeContext::NowNanoseconds() const {

        using Clock = std::chrono::steady_clock;
        auto now = Clock::now().time_since_epoch();

        return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    }

        void RuntimeContext::Log(LogSeverity severity, std::string_view message) const {
            if (!initialized_){
                std::fprintf(stderr,
                            "[RUNTIME][UNINITIALIZED][%s] %s\n",
                            Severity(severity),
                            message.empty() ? "(null)" : message.data());
                return;
            }

            std::FILE* stream = (severity == LogSeverity::kError) ? options_.error_stream
                                            : options_.info_stream;

            std::fprintf(stream, "[RUNTIME][%s] %s\n",
                        Severity(severity),
                        message.empty() ? "(null)" : message.data());
            std::fflush(stream);
        }
}

