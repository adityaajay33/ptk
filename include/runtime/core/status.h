#pragma once

#include <string>

namespace ptk
{
    namespace core
    {

        enum class StatusCode
        {
            kOk = 0,
            kInvalidArgument,
            kFailedPrecondition,
            kInternal
        };

        class Status
        {
        public:
            Status() : code_(StatusCode::kOk) {}
            Status(StatusCode code, const std::string &message) : code_(code), message_(message) {}

            static Status Ok() { return Status(); }

            bool ok() const { return code_ == StatusCode::kOk; }
            StatusCode code() const { return code_; }
            const std::string &message() const { return message_; }

        private:
            StatusCode code_;
            std::string message_;
        };

    } // namespace core
};