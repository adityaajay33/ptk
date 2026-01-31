#include "runtime/core/metrics.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace ptk::core
{

    void ComponentMetrics::ComputeStats()
    {
        if (!stage_latencies_ms.empty())
        {
            std::sort(stage_latencies_ms.begin(), stage_latencies_ms.end());
            size_t n = stage_latencies_ms.size();
            p50_latency_ms = stage_latencies_ms[n / 2];
            p95_latency_ms = stage_latencies_ms[(n * 95) / 100];
            mean_latency_ms = std::accumulate(
                                   stage_latencies_ms.begin(),
                                   stage_latencies_ms.end(), 0.0) /
                               n;
        }

        if (!frame_ages_ms.empty())
        {
            std::sort(frame_ages_ms.begin(), frame_ages_ms.end());
            size_t n = frame_ages_ms.size();
            p95_frame_age_ms = frame_ages_ms[(n * 95) / 100];
        }
    }

    void ComponentMetrics::Reset()
    {
        stage_latencies_ms.clear();
        frame_ages_ms.clear();
        frames_processed = 0;
    }

    void MetricsCollector::RecordStageLatency(const std::string &component, double latency_ms)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        component_metrics_[component].AddLatencySample(latency_ms);
    }

    void MetricsCollector::RecordFrameAge(const std::string &component, double age_ms)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        component_metrics_[component].AddFrameAgeSample(age_ms);
    }

    void MetricsCollector::IncrementFramesProcessed(const std::string &component)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        component_metrics_[component].frames_processed++;
    }

    void MetricsCollector::RecordQueueMetrics(const std::string &queue_name, const QueueStats &stats)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_metrics_[queue_name] = {
            queue_name,
            stats.current_depth,
            stats.total_pushed,
            stats.total_dropped,
            stats.total_popped};
    }

    void MetricsCollector::LogMetrics()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &[name, metrics] : component_metrics_)
        {
            metrics.ComputeStats();
        }

        std::cout << "Pipeline Metrics:" << std::endl;
        for (const auto &[name, metrics] : component_metrics_)
        {
            if (metrics.frames_processed > 0)
            {
                std::cout << "[" << name << "] Processed: " << metrics.frames_processed << " frames, "
                          << "Latency (ms): mean=" << std::fixed << std::setprecision(2) << metrics.mean_latency_ms
                          << " p50=" << metrics.p50_latency_ms
                          << " p95=" << metrics.p95_latency_ms << ", "
                          << "Frame Age p95=" << metrics.p95_frame_age_ms << " ms" << std::endl;
            }
        }

        std::cout << "Queue Metrics:" << std::endl;
        for (const auto &[name, metrics] : queue_metrics_)
        {
            double drop_rate = metrics.total_pushed > 0
                                   ? 100.0 * metrics.total_dropped / metrics.total_pushed
                                   : 0.0;
            std::cout << "[" << name << "] Depth: " << metrics.current_depth
                      << ", Drops: " << metrics.total_dropped
                      << " (" << std::fixed << std::setprecision(1) << drop_rate << "% of pushed)" << std::endl;
         }
    }

    std::string MetricsCollector::GetMetricsJson()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ostringstream json;
        json << "{\n  \"components\": {\n";

        bool first = true;
        for (auto &[name, metrics] : component_metrics_)
        {
            metrics.ComputeStats();
            if (!first)
                json << ",\n";
            first = false;

            json << "    \"" << name << "\": {\n"
                 << "      \"frames_processed\": " << metrics.frames_processed << ",\n"
                 << "      \"latency_ms\": {\n"
                 << "        \"mean\": " << metrics.mean_latency_ms << ",\n"
                 << "        \"p50\": " << metrics.p50_latency_ms << ",\n"
                 << "        \"p95\": " << metrics.p95_latency_ms << "\n"
                 << "      },\n"
                 << "      \"frame_age_p95_ms\": " << metrics.p95_frame_age_ms << "\n"
                 << "    }";
        }

        json << "\n  },\n  \"queues\": {\n";

        first = true;
        for (const auto &[name, metrics] : queue_metrics_)
        {
            if (!first)
                json << ",\n";
            first = false;

            json << "    \"" << name << "\": {\n"
                 << "      \"depth\": " << metrics.current_depth << ",\n"
                 << "      \"dropped\": " << metrics.total_dropped << ",\n"
                 << "      \"pushed\": " << metrics.total_pushed << ",\n"
                 << "      \"popped\": " << metrics.total_popped << "\n"
                 << "    }";
        }

        json << "\n  }\n}";

        return json.str();
    }

    void MetricsCollector::ResetPeriod()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[name, metrics] : component_metrics_)
        {
            metrics.Reset();
        }
    }

}