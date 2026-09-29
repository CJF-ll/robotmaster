#pragma once

#include "bounded_queue.hpp"
#include "config.hpp"
#include "error_state.hpp"
#include "types.hpp"

class ColorDetector {
 public:
  ColorDetector(const Config& config, BoundedQueue<ImageFrame>& input,
                BoundedQueue<DetectionPacket>& output, ErrorState& errors)
      : config_(config), input_(input), output_(output), errors_(errors) {}

  void run();

 private:
  TargetObservation detect(const ImageFrame& frame) const;

  const Config& config_;
  BoundedQueue<ImageFrame>& input_;
  BoundedQueue<DetectionPacket>& output_;
  ErrorState& errors_;
};
