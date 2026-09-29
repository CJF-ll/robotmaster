#include "color_detector.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

TargetObservation ColorDetector::detect(const ImageFrame& frame) const {
  TargetObservation result;
  result.sequence = frame.sequence;
  result.timestamp_us = frame.timestamp_us;

  cv::Mat hsv;
  cv::Mat mask;
  cv::cvtColor(frame.image, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(config_.h_min, config_.s_min, config_.v_min),
              cv::Scalar(config_.h_max, config_.s_max, config_.v_max), mask);

  const cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(config_.morphology_kernel, config_.morphology_kernel));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  double best_score = -1.0;
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < config_.min_area || area > config_.max_area) {
      continue;
    }

    cv::Point2f center;
    float radius = 0.0F;
    cv::minEnclosingCircle(contour, center, radius);
    if (radius <= 0.0F) {
      continue;
    }
    const double circle_area = CV_PI * radius * radius;
    const double circularity = std::clamp(area / circle_area, 0.0, 1.0);
    const double score = area * (0.5 + 0.5 * circularity);
    if (score > best_score) {
      best_score = score;
      result.detected = true;
      result.center = center;
      result.radius = radius;
      result.confidence = static_cast<float>(circularity);
    }
  }
  return result;
}

void ColorDetector::run() {
  try {
    ImageFrame frame;
    while (input_.pop(frame)) {
      DetectionPacket packet;
      packet.observation = detect(frame);
      packet.frame = std::move(frame);
      if (!output_.push(std::move(packet))) {
        break;
      }
    }
  } catch (...) {
    errors_.capture_current_exception();
    input_.close();
  }
  output_.close();
}
