#pragma once

#include <cstdint>
#include <opencv2/core.hpp>

struct ImageFrame {
  std::uint64_t sequence = 0;
  std::int64_t timestamp_us = 0;
  cv::Mat image;
};

struct TargetObservation {
  std::uint64_t sequence = 0;
  std::int64_t timestamp_us = 0;
  bool detected = false;
  cv::Point2f center{0.0F, 0.0F};
  float radius = 0.0F;
  float confidence = 0.0F;
};

struct TrackingResult {
  std::uint64_t sequence = 0;
  std::int64_t timestamp_us = 0;
  bool valid = false;
  bool predicted = false;
  cv::Point2f measured_position{0.0F, 0.0F};
  cv::Point2f filtered_position{0.0F, 0.0F};
  cv::Point2f velocity{0.0F, 0.0F};
};

struct DetectionPacket {
  ImageFrame frame;
  TargetObservation observation;
};

struct OutputPacket {
  ImageFrame frame;
  TargetObservation observation;
  TrackingResult tracking;
};
