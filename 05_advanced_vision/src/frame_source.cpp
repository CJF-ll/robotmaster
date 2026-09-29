#include "frame_source.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/videoio.hpp>

void FrameSource::run() {
  try {
    cv::VideoCapture capture(config_.input_video);
    if (!capture.isOpened()) {
      throw std::runtime_error("cannot open input video: " + config_.input_video);
    }

    const double fps = capture.get(cv::CAP_PROP_FPS);
    const std::int64_t fallback_step_us =
        static_cast<std::int64_t>(1000000.0 / ((std::isfinite(fps) && fps > 0.0) ? fps : 30.0));
    std::int64_t last_timestamp_us = -1;
    std::uint64_t sequence = 0;
    cv::Mat image;

    while (capture.read(image)) {
      std::int64_t timestamp_us = static_cast<std::int64_t>(
          std::llround(capture.get(cv::CAP_PROP_POS_MSEC) * 1000.0));
      if (timestamp_us <= last_timestamp_us) {
        timestamp_us = last_timestamp_us + std::max<std::int64_t>(1, fallback_step_us);
      }
      last_timestamp_us = timestamp_us;

      ImageFrame frame;
      frame.sequence = sequence++;
      frame.timestamp_us = timestamp_us;
      frame.image = image.clone();
      if (!output_.push(std::move(frame))) {
        break;
      }
    }
  } catch (...) {
    errors_.capture_current_exception();
  }
  output_.close();
}
