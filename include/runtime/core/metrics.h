#pragma once

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <chrono>

#include "runtime/core/queue_policy.h"

namespace ptk::core
{
    struct ComponentMetrics
    {
        std::string component_name;

        std::vector<double> stage_latencies_ms;
        double p50_latency_ms = 0.0;
        double p95_latency_ms = 0.0;
        double mean_latency_ms = 0.0;

        std::vector<double> frame_ages_ms;
        double p95_frame_age_ms = 0.0;

        uint64_t frames_processed = 0;

        void AddLatencySample(double latency_ms)
        {
            stage_latencies_ms.push_back(latency_ms);
        }

        void AddFrameAgeSample(double frame_age_ms)
        {
            frame_ages_ms.push_back(frame_age_ms);
        }

        void ComputeStats();
        void Reset();
    };

    struct QueueMetricsSnapshot
    {
        std::string queue_name;
        size_t current_depth;
        size_t total_pushed;
        size_t total_dropped;
        size_t total_popped;
    };

    class MetricsCollector
    {

    public:
        static MetricsCollector &Instance()
        {
            static MetricsCollector instance;
            return instance;
        }

        void RecordStageLatency(const std::string &component, double latency_ms);
        void RecordFrameAge(const std::string &component, double age_ms);
        void IncrementFramesProcessed(const std::string &component);

        void RecordQueueMetrics(const std::string &queue_name, const QueueStats &stats);

        void LogMetrics();          
        std::string GetMetricsJson(); 
        void ResetPeriod();

    private:
        MetricsCollector() = default;
        mutable std::mutex mutex_;
        std::map<std::string, ComponentMetrics> component_metrics_;
        std::map<std::string, QueueMetricsSnapshot> queue_metrics_;
    };
}