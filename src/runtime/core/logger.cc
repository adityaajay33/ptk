#include "runtime/core/logger.h"
#include <iomanip>
#include <mutex>

namespace ptk::core
{

    // ConsoleLogger Implementation

    ConsoleLogger::ConsoleLogger(LogLevel min_level)
        : min_level_(min_level), use_colors_(true) {}

    void ConsoleLogger::Log(LogLevel level, const std::string &category,
                            const std::string &message)
    {
        if (static_cast<int>(level) < static_cast<int>(min_level_))
        {
            return;
        }

        std::ostringstream oss;

        // Add timestamp
        oss << GetTimestamp();

        // Add color if enabled
        if (use_colors_)
        {
            oss << GetColorCode(level);
        }

        // Format: [TIME] [LEVEL] [CATEGORY] message
        oss << " [" << GetLevelString(level) << "] [" << category << "] "
            << message;

        // Reset color
        if (use_colors_)
        {
            oss << GetResetCode();
        }

        oss << "\n";

        // Write to stderr for warnings/errors, stdout for others
        std::ostream &out = (level >= LogLevel::kWarn) ? std::cerr : std::cout;
        out << oss.str() << std::flush;
    }

    std::string ConsoleLogger::GetLevelString(LogLevel level) const
    {
        switch (level)
        {
        case LogLevel::kDebug:
            return "DEBUG";
        case LogLevel::kInfo:
            return "INFO";
        case LogLevel::kWarn:
            return "WARN";
        case LogLevel::kError:
            return "ERROR";
        case LogLevel::kCritical:
            return "CRIT";
        case LogLevel::kSilent:
            return "SILENT";
        default:
            return "UNKNOWN";
        }
    }

    std::string ConsoleLogger::GetColorCode(LogLevel level) const
    {
#ifdef _WIN32
        // Windows console doesn't support ANSI codes by default
        return "";
#else
        switch (level)
        {
        case LogLevel::kDebug:
            return "\033[36m"; // Cyan
        case LogLevel::kInfo:
            return "\033[32m"; // Green
        case LogLevel::kWarn:
            return "\033[33m"; // Yellow
        case LogLevel::kError:
            return "\033[31m"; // Red
        case LogLevel::kCritical:
            return "\033[35m"; // Magenta
        default:
            return "";
        }
#endif
    }

    std::string ConsoleLogger::GetResetCode() const
    {
#ifdef _WIN32
        return "";
#else
        return "\033[0m";
#endif
    }

    std::string ConsoleLogger::GetTimestamp() const
    {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t_now), "%H:%M:%S");

        // Add milliseconds
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();

        return oss.str();
    }

    // LoggerManager Implementation

    std::shared_ptr<Logger> LoggerManager::global_logger_;

    LoggerManager &LoggerManager::GetInstance()
    {
        static LoggerManager instance;
        return instance;
    }

    LoggerManager::LoggerManager()
    {
        // Initialize with default console logger
        if (!global_logger_)
        {
            global_logger_ =
                std::make_shared<ConsoleLogger>(LogLevel::kInfo);
        }
    }

    void LoggerManager::SetLogger(std::shared_ptr<Logger> logger)
    {
        GetInstance();
        global_logger_ = logger;
    }

    std::shared_ptr<Logger> LoggerManager::GetLogger()
    {
        GetInstance();
        return global_logger_;
    }

    void LoggerManager::Log(LogLevel level, const std::string &category,
                            const std::string &message)
    {
        auto logger = GetLogger();
        if (logger)
        {
            logger->Log(level, category, message);
        }
    }

    void LoggerManager::Debug(const std::string &category,
                              const std::string &msg)
    {
        Log(LogLevel::kDebug, category, msg);
    }

    void LoggerManager::Info(const std::string &category, const std::string &msg)
    {
        Log(LogLevel::kInfo, category, msg);
    }

    void LoggerManager::Warn(const std::string &category, const std::string &msg)
    {
        Log(LogLevel::kWarn, category, msg);
    }

    void LoggerManager::Error(const std::string &category,
                              const std::string &msg)
    {
        Log(LogLevel::kError, category, msg);
    }

    void LoggerManager::Critical(const std::string &category,
                                 const std::string &msg)
    {
        Log(LogLevel::kCritical, category, msg);
    }

    void LoggerManager::SetGlobalLogLevel(LogLevel level)
    {
        auto logger = GetLogger();
        if (logger)
        {
            logger->SetLogLevel(level);
        }
    }

} // namespace ptk::core
