#include "target_tracker.hpp"

#include <algorithm>
#include <cmath>

TrackingResult TargetTracker::update(const TargetObservation& observation) {
  TrackingResult result;
  result.sequence = observation.sequence;
  result.timestamp_us = observation.timestamp_us;
  result.measured_position = observation.center;

  if (!initialized_) {
    if (observation.detected) {
      initialized_ = true;
      position_ = observation.center;
      velocity_ = cv::Point2f(0.0F, 0.0F);
      last_timestamp_us_ = observation.timestamp_us;
      missed_frames_ = 0;
      result.valid = true;
      result.filtered_position = position_;
      result.velocity = velocity_;
    }
    return result;
  }

  double dt = static_cast<double>(observation.timestamp_us - last_timestamp_us_) / 1000000.0;
  dt = std::clamp(dt, 0.001, 1.0);
  last_timestamp_us_ = observation.timestamp_us;
  const cv::Point2f predicted_position = position_ + velocity_ * static_cast<float>(dt);

  bool accepted_measurement = false;
  if (observation.detected) {
    const cv::Point2f residual = observation.center - predicted_position;
    const double distance = std::hypot(residual.x, residual.y);
    if (distance <= config_.max_match_distance) {
      position_ = predicted_position +
                  residual * static_cast<float>(config_.tracker_alpha);
      velocity_ += residual * static_cast<float>(config_.tracker_beta / dt);
      missed_frames_ = 0;
      accepted_measurement = true;
    }
  }

  if (!accepted_measurement) {
    position_ = predicted_position;
    ++missed_frames_;
    if (missed_frames_ > config_.max_missed_frames) {
      initialized_ = false;
      velocity_ = cv::Point2f(0.0F, 0.0F);
      return result;
    }
  }

  result.valid = true;
  result.predicted = !accepted_measurement;
  result.filtered_position = position_;
  result.velocity = velocity_;
  return result;
}

void TargetTracker::run() {
  try {
    DetectionPacket packet;
    while (input_.pop(packet)) {
      OutputPacket output_packet;
      output_packet.tracking = update(packet.observation);
      output_packet.observation = packet.observation;
      output_packet.frame = std::move(packet.frame);
      if (!output_.push(std::move(output_packet))) {
        break;
      }
    }
  } catch (...) {
    errors_.capture_current_exception();
    input_.close();
  }
  output_.close();
}
