#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>

enum class SensorType { CAMERA, IMU };
struct SensorData { SensorType type; std::uint32_t sequence; long long timestamp_us; };
struct SyncResult { SensorData camera; std::optional<SensorData> imu; long long delta_us = 0; };
inline bool timestamp_sequence_less(const SensorData& a, const SensorData& b) {
  return a.timestamp_us != b.timestamp_us ? a.timestamp_us < b.timestamp_us : a.sequence < b.sequence;
}
inline std::vector<SyncResult> synchronize(std::vector<SensorData> cameras,
                                           std::vector<SensorData> imus,
                                           long long max_delta_us) {
  if (max_delta_us < 0) throw std::invalid_argument("max_delta_us must be greater than or equal to zero");
  std::sort(cameras.begin(), cameras.end(), timestamp_sequence_less);
  std::sort(imus.begin(), imus.end(), timestamp_sequence_less);
  std::vector<SensorData> unique_imus;
  for (const auto& imu : imus) {
    if (unique_imus.empty() || unique_imus.back().timestamp_us != imu.timestamp_us) unique_imus.push_back(imu);
  }
  std::vector<SyncResult> results;
  std::size_t right = 0;
  for (const auto& camera : cameras) {
    while (right < unique_imus.size() && unique_imus[right].timestamp_us < camera.timestamp_us) ++right;
    const SensorData* best = nullptr;
    long long best_delta = 0;
    auto consider = [&](const SensorData& imu) {
      const long long delta = std::llabs(camera.timestamp_us - imu.timestamp_us);
      if (!best || delta < best_delta ||
          (delta == best_delta && imu.timestamp_us < best->timestamp_us) ||
          (delta == best_delta && imu.timestamp_us == best->timestamp_us && imu.sequence < best->sequence)) {
        best = &imu;
        best_delta = delta;
      }
    };
    if (right < unique_imus.size()) consider(unique_imus[right]);
    if (right > 0) consider(unique_imus[right - 1]);
    if (best && best_delta <= max_delta_us) results.push_back({camera, *best, best_delta});
    else results.push_back({camera, std::nullopt, 0});
  }
  return results;
}
