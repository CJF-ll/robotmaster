#pragma once

#include <optional>

#include "config.hpp"
#include "types.hpp"

std::optional<ArmorObservation> build_armor_observation(
    const LightBar& first, const LightBar& second, const Config& config, cv::Size image_size);

class ArmorDetector {
 public:
  explicit ArmorDetector(const Config& config) : config_(config) {}
  DetectionFrame detect(const cv::Mat& bgr, cv::Mat* mask_out = nullptr) const;

 private:
  Config config_;
};
