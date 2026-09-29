#pragma once

#include "bounded_queue.hpp"
#include "config.hpp"
#include "error_state.hpp"
#include "types.hpp"

class FrameSource {
 public:
  FrameSource(const Config& config, BoundedQueue<ImageFrame>& output,
              ErrorState& errors)
      : config_(config), output_(output), errors_(errors) {}

  void run();

 private:
  const Config& config_;
  BoundedQueue<ImageFrame>& output_;
  ErrorState& errors_;
};
