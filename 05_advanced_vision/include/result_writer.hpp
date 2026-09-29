#pragma once

#include "bounded_queue.hpp"
#include "config.hpp"
#include "error_state.hpp"
#include "types.hpp"

class ResultWriter {
 public:
  ResultWriter(const Config& config, BoundedQueue<OutputPacket>& input,
               ErrorState& errors)
      : config_(config), input_(input), errors_(errors) {}

  void run();

 private:
  const Config& config_;
  BoundedQueue<OutputPacket>& input_;
  ErrorState& errors_;
};
