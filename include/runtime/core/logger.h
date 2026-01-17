#pragma once

#include "runtime/core/status.h"
#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <memory>

namespace ptk::core {

// ============================================================================
// Extended Status Codes for Inference Engines
// ============================================================================

enum class EngineErrorCode {
    // Success
    kOk = 0,
    
    // Model Loading Errors
    kModelNotFound = 100,
    kModelLoadFailed = 101,
    kModelInvalid = 102,
    kUnsupportedModelFormat = 103,
    
    // Execution Provider Errors
    kExecutionProviderNotAvailable = 200,
    kExecutionProviderInitFailed = 201,
    kCudaNotAvailable = 202,
    kTensorRTNotAvailable = 203,
    
    // Tensor/Shape Errors
    kShapeMismatch = 300,
    kNameMismatch = 301,
    kTypeMismatch = 302,
    kDimensionMismatch = 303,
    
    // Inference Errors
    kInferenceFailed = 400,
    kInferenceTimeout = 401,
    kMemoryAllocationFailed = 402,
    kDeviceMemoryFull = 403,
    kSessionNotLoaded = 404,
    
    // Configuration Errors
    kInvalidConfig = 500,
    kConfigConflict = 501,
    kUnsupportedConfig = 502,
    
    // Generic/Unknown
    kInternalError = 600,
    kUnknownError = 601
};

// ============================================================================
// Log Levels
// ============================================================================

enum class LogLevel {
    kDebug = 0,
    kInfo = 1,
    kWarn = 2,
    kError = 3,
    kCritical = 4,
    kSilent = 5
};

// ============================================================================
// Logger Interface
// ============================================================================

class Logger {
public:
    virtual ~Logger() = default;

    // Log a message with level
    virtual void Log(LogLevel level, const std::string& category,
                     const std::string& message) = 0;

    // Convenience methods
    void Debug(const std::string& category, const std::string& msg) {
        Log(LogLevel::kDebug, category, msg);
    }
    void Info(const std::string& category, const std::string& msg) {
        Log(LogLevel::kInfo, category, msg);
    }
    void Warn(const std::string& category, const std::string& msg) {
        Log(LogLevel::kWarn, category, msg);
    }
    void Error(const std::string& category, const std::string& msg) {
        Log(LogLevel::kError, category, msg);
    }
    void Critical(const std::string& category, const std::string& msg) {
        Log(LogLevel::kCritical, category, msg);
    }

    // Set log level
    virtual void SetLogLevel(LogLevel level) = 0;
    virtual LogLevel GetLogLevel() const = 0;

    // Enable/disable colors (for console output)
    virtual void SetColorEnabled(bool enabled) = 0;
};

// ============================================================================
// Console Logger Implementation
// ============================================================================

class ConsoleLogger : public Logger {
public:
    explicit ConsoleLogger(LogLevel min_level = LogLevel::kInfo);
    ~ConsoleLogger() override = default;

    void Log(LogLevel level, const std::string& category,
             const std::string& message) override;

    void SetLogLevel(LogLevel level) override { min_level_ = level; }
    LogLevel GetLogLevel() const override { return min_level_; }

    void SetColorEnabled(bool enabled) override { use_colors_ = enabled; }

private:
    LogLevel min_level_;
    bool use_colors_;

    std::string GetLevelString(LogLevel level) const;
    std::string GetColorCode(LogLevel level) const;
    std::string GetResetCode() const;
    std::string GetTimestamp() const;
};

// ============================================================================
// Global Logger Management
// ============================================================================

class LoggerManager {
public:
    static LoggerManager& GetInstance();

    // Set the global logger
    static void SetLogger(std::shared_ptr<Logger> logger);

    // Get the global logger
    static std::shared_ptr<Logger> GetLogger();

    // Quick logging
    static void Log(LogLevel level, const std::string& category,
                    const std::string& message);

    static void Debug(const std::string& category, const std::string& msg);
    static void Info(const std::string& category, const std::string& msg);
    static void Warn(const std::string& category, const std::string& msg);
    static void Error(const std::string& category, const std::string& msg);
    static void Critical(const std::string& category, const std::string& msg);

    // Set global minimum log level
    static void SetGlobalLogLevel(LogLevel level);

private:
    LoggerManager();
    static std::shared_ptr<Logger> global_logger_;
};

// ============================================================================
// Extended Status Class with Error Codes
// ============================================================================

class EngineStatus {
public:
    EngineStatus() : code_(StatusCode::kOk), error_code_(EngineErrorCode::kOk) {}
    
    EngineStatus(StatusCode code, const std::string& message)
        : code_(code), message_(message), error_code_(EngineErrorCode::kOk) {}
    
    EngineStatus(EngineErrorCode error_code, const std::string& message)
        : code_(error_code == EngineErrorCode::kOk ? StatusCode::kOk
                                                      : StatusCode::kInternal),
          message_(message),
          error_code_(error_code) {}

    static EngineStatus Ok() { return EngineStatus(); }
    
    static EngineStatus ModelNotFound(const std::string& path) {
        return EngineStatus(EngineErrorCode::kModelNotFound,
                           "Model not found: " + path);
    }
    
    static EngineStatus ModelLoadFailed(const std::string& reason) {
        return EngineStatus(EngineErrorCode::kModelLoadFailed,
                           "Failed to load model: " + reason);
    }
    
    static EngineStatus ShapeMismatch(const std::string& expected,
                                      const std::string& actual) {
        return EngineStatus(EngineErrorCode::kShapeMismatch,
                           "Shape mismatch - expected: " + expected +
                               ", got: " + actual);
    }
    
    static EngineStatus ExecutionProviderNotAvailable(const std::string& ep) {
        return EngineStatus(EngineErrorCode::kExecutionProviderNotAvailable,
                           "Execution provider not available: " + ep);
    }
    
    static EngineStatus InferenceFailed(const std::string& reason) {
        return EngineStatus(EngineErrorCode::kInferenceFailed,
                           "Inference failed: " + reason);
    }

    bool ok() const { return code_ == StatusCode::kOk; }
    StatusCode code() const { return code_; }
    EngineErrorCode error_code() const { return error_code_; }
    const std::string& message() const { return message_; }

private:
    StatusCode code_;
    std::string message_;
    EngineErrorCode error_code_;
};

}  // namespace ptk::core
