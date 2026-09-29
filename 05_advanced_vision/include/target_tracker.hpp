#pragma once

#include "bounded_queue.hpp"
#include "config.hpp"
#include "error_state.hpp"
#include "types.hpp"

class TargetTracker {
 public:
  TargetTracker(const Config& config, BoundedQueue<DetectionPacket>& input,
                BoundedQueue<OutputPacket>& output, ErrorState& errors)
      : config_(config), input_(input), output_(output), errors_(errors) {}

  void run();

 private:
  TrackingResult update(const TargetObservation& observation);

  const Config& config_;
  BoundedQueue<DetectionPacket>& input_;
  BoundedQueue<OutputPacket>& output_;
  ErrorState& errors_;
  bool initialized_ = false;
  cv::Point2f position_{0.0F, 0.0F};
  cv::Point2f velocity_{0.0F, 0.0F};
  std::int64_t last_timestamp_us_ = 0;
  int missed_frames_ = 0;
};
