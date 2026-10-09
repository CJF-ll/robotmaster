#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "armor_detector.hpp"
#include "config.hpp"
#include "render_policy.hpp"
#include "rigid_model.hpp"
#include "types.hpp"

namespace {
class FramePreprocessor {
 public:
  FramePreprocessor(const Config& config, cv::Size input_size) : enabled_(config.camera_enabled) {
    if (!enabled_) return;
    const double sx = static_cast<double>(input_size.width) / config.camera_reference_width;
    const double sy = static_cast<double>(input_size.height) / config.camera_reference_height;
    camera_matrix_ = (cv::Mat_<double>(3, 3) << config.fx * sx, 0, config.cx * sx,
                      0, config.fy * sy, config.cy * sy, 0, 0, 1);
    distortion_ = (cv::Mat_<double>(1, 5) << config.k1, config.k2, config.p1, config.p2, config.k3);
  }

  cv::Mat process(const cv::Mat& input) const {
    if (!enabled_) return input;
    cv::Mat output;
    cv::undistort(input, output, camera_matrix_, distortion_);
    return output;
  }

 private:
  bool enabled_ = false;
  cv::Mat camera_matrix_, distortion_;
};

std::string mode_name(MotionMode mode) {
  switch (mode) {
    case MotionMode::kStopped: return "STOPPED";
    case MotionMode::kUniform: return "UNIFORM";
    case MotionMode::kAccelerating: return "ACCELERATING";
    case MotionMode::kDecelerating: return "DECELERATING";
    case MotionMode::kReversing: return "REVERSING";
    case MotionMode::kLost: return "LOST";
    default: return "INITIALIZING";
  }
}

cv::Mat motion_signature(const cv::Mat& frame, const Config& config) {
  const int x0 = cvRound(config.roi_x_min * frame.cols);
  const int y0 = cvRound(config.roi_y_min * frame.rows);
  const int x1 = cvRound(config.roi_x_max * frame.cols);
  const int y1 = cvRound(config.roi_y_max * frame.rows);
  cv::Mat gray, signature;
  cv::cvtColor(frame(cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0))),
               gray, cv::COLOR_BGR2GRAY);
  cv::resize(gray, signature, {160, 96}, 0, 0, cv::INTER_AREA);
  return signature;
}

double scalar_median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

ArmorObservation median_observation(const std::vector<ArmorObservation>& samples) {
  ArmorObservation output;
  if (samples.empty()) return output;
  std::vector<double> center_x, center_y, widths, heights, scores;
  std::array<std::vector<double>, 4> corner_x, corner_y;
  center_x.reserve(samples.size());
  center_y.reserve(samples.size());
  for (const auto& sample : samples) {
    center_x.push_back(sample.center.x);
    center_y.push_back(sample.center.y);
    widths.push_back(sample.size.width);
    heights.push_back(sample.size.height);
    scores.push_back(sample.score);
    for (int corner = 0; corner < 4; ++corner) {
      corner_x[corner].push_back(sample.corners[corner].x);
      corner_y[corner].push_back(sample.corners[corner].y);
    }
  }
  output.center = {static_cast<float>(scalar_median(center_x)),
                   static_cast<float>(scalar_median(center_y))};
  output.size = {static_cast<float>(scalar_median(widths)),
                 static_cast<float>(scalar_median(heights))};
  output.score = static_cast<float>(scalar_median(scores));
  for (int corner = 0; corner < 4; ++corner) {
    output.corners[corner] = {
        static_cast<float>(scalar_median(corner_x[corner])),
        static_cast<float>(scalar_median(corner_y[corner]))};
  }
  return output;
}

struct OfflineHandover {
  int frame = -1;
  int source_slot = -1;
  ArmorObservation exit;
  ArmorObservation entry;
};

struct HandoverTrainingSample {
  int age_frames = 0;
  double progress = 0.0;
  bool positive = false;
};

void fit_handover_age_gate(const std::vector<HandoverTrainingSample>& samples,
                           int maximum_age,
                           MotionPrior::HandoverSlotPrior* output) {
  if (samples.empty()) return;
  int best_errors = std::numeric_limits<int>::max();
  int best_false_positives = std::numeric_limits<int>::max();
  int best_false_negatives = std::numeric_limits<int>::max();
  int best_age = 0;
  for (int age = 0; age <= maximum_age; ++age) {
    int false_positives = 0, false_negatives = 0;
    for (const auto& sample : samples) {
      const bool prediction = sample.age_frames >= age;
      false_positives += prediction && !sample.positive;
      false_negatives += !prediction && sample.positive;
    }
    const int errors = false_positives + false_negatives;
    if (errors < best_errors ||
        (errors == best_errors && false_positives < best_false_positives) ||
        (errors == best_errors && false_positives == best_false_positives &&
         false_negatives < best_false_negatives) ||
        (errors == best_errors && false_positives == best_false_positives &&
         false_negatives == best_false_negatives && age > best_age)) {
      best_errors = errors;
      best_false_positives = false_positives;
      best_false_negatives = false_negatives;
      best_age = age;
    }
  }
  output->age_threshold_frames = best_age;
}

void fit_handover_score_gate(
    const std::vector<HandoverTrainingSample>& samples, const Config& config,
    MotionPrior::HandoverSlotPrior* output) {
  if (samples.empty()) return;
  int best_errors = std::numeric_limits<int>::max();
  int best_false_positives = std::numeric_limits<int>::max();
  int best_false_negatives = std::numeric_limits<int>::max();
  double best_weight = 0.0;
  double best_threshold = 0.0;
  const int weight_steps = std::max(
      0, cvRound(config.image_prediction_handover_age_weight_max_px /
                 config.image_prediction_handover_age_weight_step_px));
  for (int weight_index = 0; weight_index <= weight_steps; ++weight_index) {
    const double weight = weight_index *
        config.image_prediction_handover_age_weight_step_px;
    std::vector<double> observed_scores;
    observed_scores.reserve(samples.size());
    for (const auto& sample : samples) {
      observed_scores.push_back(
          sample.progress - weight * sample.age_frames);
    }
    std::sort(observed_scores.begin(), observed_scores.end());
    observed_scores.erase(
        std::unique(observed_scores.begin(), observed_scores.end()),
        observed_scores.end());
    std::vector<double> thresholds;
    thresholds.reserve(observed_scores.size() + 1);
    thresholds.push_back(observed_scores.front() - 1.0);
    for (std::size_t index = 1; index < observed_scores.size(); ++index) {
      thresholds.push_back(0.5 * (observed_scores[index - 1] +
                                  observed_scores[index]));
    }
    thresholds.push_back(observed_scores.back() + 1.0);
    for (double threshold : thresholds) {
      int false_positives = 0, false_negatives = 0;
      for (const auto& sample : samples) {
        const bool prediction =
            sample.progress - weight * sample.age_frames <= threshold;
        false_positives += prediction && !sample.positive;
        false_negatives += !prediction && sample.positive;
      }
      const int errors = false_positives + false_negatives;
      const bool better = errors < best_errors ||
          (errors == best_errors && false_positives < best_false_positives) ||
          (errors == best_errors && false_positives == best_false_positives &&
           false_negatives < best_false_negatives) ||
          (errors == best_errors && false_positives == best_false_positives &&
           false_negatives == best_false_negatives && weight < best_weight);
      if (better) {
        best_errors = errors;
        best_false_positives = false_positives;
        best_false_negatives = false_negatives;
        best_weight = weight;
        best_threshold = threshold;
      }
    }
  }
  output->progress_age_weight_px = best_weight;
  output->progress_threshold = best_threshold;
}

void calibrate_handover_prior(
    const std::vector<std::optional<ArmorObservation>>& observations,
    const AffineGeometry& geometry, const Config& config, MotionPrior* prior) {
  if (!config.image_prediction_handover_prior_enabled || observations.empty()) return;
  std::vector<OfflineHandover> transitions;
  int previous_frame = -1;
  for (int frame = 0; frame < static_cast<int>(observations.size()); ++frame) {
    if (!observations[frame].has_value()) continue;
    if (previous_frame >= 0) {
      const int frame_gap = frame - previous_frame;
      const double step = cv::norm(observations[frame]->center -
                                   observations[previous_frame]->center) /
                          std::max(1, frame_gap);
      if (step >= config.image_prediction_jump_min_px &&
          step <= config.image_prediction_jump_max_px) {
        transitions.push_back({frame, -1, *observations[previous_frame],
                               *observations[frame]});
      }
    }
    previous_frame = frame;
  }
  if (transitions.size() < static_cast<std::size_t>(
          3 * config.image_prediction_handover_min_transitions_per_slot)) return;

  std::vector<double> direction_x, direction_y;
  std::vector<double> image_direction_x, image_direction_y;
  for (const auto& transition : transitions) {
    const cv::Point2d delta =
        geometry.normalized_coordinates(transition.entry.center) -
        geometry.normalized_coordinates(transition.exit.center);
    direction_x.push_back(delta.x);
    direction_y.push_back(delta.y);
    const cv::Point2d image_delta =
        cv::Point2d(transition.entry.center) -
        cv::Point2d(transition.exit.center);
    image_direction_x.push_back(image_delta.x);
    image_direction_y.push_back(image_delta.y);
  }
  cv::Point2d forward(scalar_median(direction_x), scalar_median(direction_y));
  const double forward_length = cv::norm(forward);
  if (forward_length < 1e-6) return;
  forward *= 1.0 / forward_length;
  cv::Point2d image_forward(scalar_median(image_direction_x),
                            scalar_median(image_direction_y));
  const double image_forward_length = cv::norm(image_forward);
  if (image_forward_length < 1e-6) return;
  image_forward *= 1.0 / image_forward_length;

  std::vector<OfflineHandover> dominant_transitions;
  dominant_transitions.reserve(transitions.size());
  int current_slot = 0;
  for (auto transition : transitions) {
    const cv::Point2d delta =
        geometry.normalized_coordinates(transition.entry.center) -
        geometry.normalized_coordinates(transition.exit.center);
    if (delta.dot(forward) <= 0.0) continue;
    transition.source_slot = current_slot;
    dominant_transitions.push_back(transition);
    current_slot = (current_slot + 1) % 3;
  }
  if (dominant_transitions.size() < static_cast<std::size_t>(
          3 * config.image_prediction_handover_min_transitions_per_slot)) return;

  prior->handover.forward_direction_normalized = forward;
  prior->handover.forward_direction_image = image_forward;
  prior->handover.first_transition_frame = dominant_transitions.front().frame;
  prior->handover.last_active_frame = prior->detected_motion_end_frame;
  std::array<std::vector<ArmorObservation>, 3> exits, entries;
  std::vector<int> transition_frames;
  std::vector<int> transition_index_by_frame(observations.size(), -1);
  for (int index = 0; index < static_cast<int>(dominant_transitions.size()); ++index) {
    const auto& transition = dominant_transitions[index];
    exits[transition.source_slot].push_back(transition.exit);
    entries[transition.source_slot].push_back(transition.entry);
    transition_frames.push_back(transition.frame);
    transition_index_by_frame[transition.frame] = index;
  }

  std::vector<int> slots(observations.size(), 0), ages(observations.size(), 0);
  current_slot = 0;
  int age = 0;
  for (int frame = 0; frame < static_cast<int>(observations.size()); ++frame) {
    if (transition_index_by_frame[frame] >= 0) {
      current_slot = (current_slot + 1) % 3;
      age = 0;
    } else if (frame > 0) {
      ++age;
    }
    slots[frame] = current_slot;
    ages[frame] = age;
  }

  std::array<std::vector<HandoverTrainingSample>, 3> training;
  const int first_frame = std::max(0, prior->detected_motion_start_frame);
  const int last_frame = std::min(
      static_cast<int>(observations.size()) - 1, prior->detected_motion_end_frame);
  for (int frame = first_frame; frame <= last_frame; ++frame) {
    if (!observations[frame].has_value()) continue;
    const auto next = std::upper_bound(
        transition_frames.begin(), transition_frames.end(), frame);
    const bool positive = next != transition_frames.end() &&
        *next <= frame + config.prediction_lead_frames;
    const double progress = cv::Point2d(observations[frame]->center)
        .dot(image_forward);
    training[slots[frame]].push_back({ages[frame], progress, positive});
  }

  bool all_slots_valid = true;
  for (int slot = 0; slot < 3; ++slot) {
    auto& slot_prior = prior->handover.slots[slot];
    slot_prior.transition_samples = static_cast<int>(exits[slot].size());
    slot_prior.valid = slot_prior.transition_samples >=
        config.image_prediction_handover_min_transitions_per_slot &&
        !training[slot].empty();
    if (!slot_prior.valid) {
      all_slots_valid = false;
      continue;
    }
    slot_prior.exit_template = median_observation(exits[slot]);
    slot_prior.entry_template = median_observation(entries[slot]);
    fit_handover_age_gate(
        training[slot], config.image_prediction_handover_training_max_age_frames,
        &slot_prior);
    fit_handover_score_gate(training[slot], config, &slot_prior);
  }
  prior->handover.valid = all_slots_valid;
}

MotionPrior estimate_motion_prior(const std::string& input_path, const Config& config,
                                  const AffineGeometry& geometry, cv::Size image_size,
                                  double fps) {
  cv::VideoCapture capture(input_path);
  if (!capture.isOpened()) throw std::runtime_error("cannot reopen video for temporal calibration");
  FramePreprocessor preprocessor(config, image_size);
  ArmorDetector detector(config);
  std::vector<cv::Mat> signatures;
  std::vector<double> raw_phases;
  std::vector<std::optional<ArmorObservation>> observations;
  cv::Mat raw;
  while (capture.read(raw)) {
    cv::Mat frame = preprocessor.process(raw);
    signatures.push_back(motion_signature(frame, config));
    const DetectionFrame detections = detector.detect(frame);
    if (detections.armors.empty()) {
      observations.push_back(std::nullopt);
      raw_phases.push_back(std::numeric_limits<double>::quiet_NaN());
    } else {
      observations.push_back(detections.armors.front());
      raw_phases.push_back(geometry.phase_of(detections.armors.front().center));
    }
  }
  MotionPrior prior;
  prior.calibration_fps = fps;
  if (signatures.size() < 10) return prior;
  std::vector<double> energy(signatures.size(), 0.0);
  for (std::size_t i = 1; i < signatures.size(); ++i)
    energy[i] = cv::norm(signatures[i], signatures[i - 1], cv::NORM_L1) /
                static_cast<double>(signatures[i].total());
  const int hold = std::max(1, config.motion_hold_frames);
  for (int i = hold; i < static_cast<int>(energy.size()); ++i) {
    bool active = true;
    for (int k = 0; k < hold; ++k)
      active = active && energy[i - k] > config.motion_difference_threshold;
    if (active) { prior.detected_motion_start_frame = i - hold + 1; break; }
  }
  for (int i = static_cast<int>(energy.size()) - 1 - hold; i >= 1; --i) {
    bool active = true;
    for (int k = 0; k < hold; ++k)
      active = active && energy[i + k] > config.motion_difference_threshold;
    if (active) { prior.detected_motion_end_frame = i + hold - 1; break; }
  }
  if (prior.detected_motion_start_frame < 0 ||
      prior.detected_motion_end_frame <= prior.detected_motion_start_frame + 30)
    return prior;

  const int minimum_lag = std::max(2, cvRound(config.period_search_min_s * fps));
  const int maximum_lag = std::min(cvRound(config.period_search_max_s * fps),
                                   prior.detected_motion_end_frame - prior.detected_motion_start_frame - 1);
  double best_score = std::numeric_limits<double>::infinity();
  for (int lag = minimum_lag; lag <= maximum_lag; ++lag) {
    double score = 0.0;
    int count = 0;
    for (int i = prior.detected_motion_start_frame + 10;
         i + lag <= prior.detected_motion_end_frame - 10; i += 2) {
      score += cv::norm(signatures[i], signatures[i + lag], cv::NORM_L1) /
               static_cast<double>(signatures[i].total());
      ++count;
    }
    if (count > 0 && score / count < best_score) {
      best_score = score / count;
      prior.repeat_period_frames = lag;
    }
  }

  std::vector<double> phase_steps;
  for (int i = std::max(1, prior.detected_motion_start_frame);
       i <= prior.detected_motion_end_frame; ++i) {
    if (!std::isfinite(raw_phases[i]) || !std::isfinite(raw_phases[i - 1])) continue;
    const double step = std::remainder(raw_phases[i] - raw_phases[i - 1], 2.0 * CV_PI / 3.0);
    if (std::abs(step) < 0.35) phase_steps.push_back(step);
  }
  const double median_step = scalar_median(phase_steps);
  if (prior.repeat_period_frames > 0 && std::abs(median_step) > 1e-4) {
    prior.speed_abs_rad_s = (2.0 * CV_PI / 3.0) * fps / prior.repeat_period_frames;
    prior.phase_direction_sign = median_step > 0.0 ? 1 : -1;
    prior.valid = true;
  }
  calibrate_handover_prior(observations, geometry, config, &prior);
  return prior;
}

class OnlineMotionDetector {
 public:
  explicit OnlineMotionDetector(const Config& config) : config_(config) {}
  bool update(const cv::Mat& frame) {
    cv::Mat current = motion_signature(frame, config_);
    if (previous_.empty()) { previous_ = current; return false; }
    const double energy = cv::norm(current, previous_, cv::NORM_L1) /
                          static_cast<double>(current.total());
    previous_ = current;
    if (energy > config_.motion_difference_threshold) {
      positive_ = std::min(positive_ + 1, config_.motion_hold_frames);
      negative_ = 0;
      if (positive_ >= config_.motion_hold_frames) moving_ = true;
    } else {
      negative_ = std::min(negative_ + 1, config_.motion_hold_frames);
      positive_ = 0;
      if (negative_ >= config_.motion_hold_frames) moving_ = false;
    }
    return moving_;
  }
 private:
  Config config_;
  cv::Mat previous_;
  int positive_ = 0, negative_ = 0;
  bool moving_ = false;
};

void draw_quad(cv::Mat& image, const std::array<cv::Point2f, 4>& corners,
               const cv::Scalar& color, int thickness) {
  for (int i = 0; i < 4; ++i)
    cv::line(image, corners[i], corners[(i + 1) % 4], color, thickness, cv::LINE_AA);
}

void draw_dashed_quad(cv::Mat& image, const std::array<cv::Point2f, 4>& corners,
                      const cv::Scalar& color, int thickness) {
  constexpr float kDashLength = 6.0F;
  constexpr float kGapLength = 4.0F;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f start = corners[i];
    const cv::Point2f delta = corners[(i + 1) % 4] - start;
    const float length = static_cast<float>(cv::norm(delta));
    if (length < 1.0F) continue;
    const cv::Point2f direction = delta * (1.0F / length);
    for (float offset = 0.0F; offset < length; offset += kDashLength + kGapLength) {
      const float dash_end = std::min(length, offset + kDashLength);
      cv::line(image, start + direction * offset, start + direction * dash_end,
               color, thickness, cv::LINE_AA);
    }
  }
}

std::string source_name(BoxSource source) {
  switch (source) {
    case BoxSource::kObserved: return "current_detection";
    case BoxSource::kPredicted: return "current_model";
    default: return "none";
  }
}

void draw_overlay(cv::Mat& frame, const SolverOutput& state, int frame_number,
                  const Config& config,
                  const PredictionRenderDecision& prediction_render,
                  bool draw_labels, bool debug_lights) {
  for (const auto& slot : state.slots) {
    if (!slot.valid || slot.source == BoxSource::kNone) continue;
    const bool detected_now = slot.source == BoxSource::kObserved;
    const cv::Scalar color(0, 255, 0);
    if (detected_now) {
      draw_quad(frame, slot.corners, color, 2);
      cv::circle(frame, slot.center, 2, color, cv::FILLED);
    } else {
      draw_dashed_quad(frame, slot.corners, color, 1);
    }
    if (draw_labels) {
      const std::string label = "A" + std::to_string(slot.id) +
          (detected_now ? " NOW" : "");
      cv::putText(frame, label, slot.center + cv::Point2f(4, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv::LINE_AA);
    }
  }

  if (prediction_render.draw) {
    const cv::Scalar future_color(0, 0, 255);
    if (prediction_render.high_overlap) {
      cv::Mat overlay = frame.clone();
      draw_dashed_quad(overlay, state.future_target.corners, future_color, 1);
      cv::circle(overlay, state.future_target.center, 3, future_color, 1,
                 cv::LINE_AA);
      cv::addWeighted(overlay, config.prediction_overlap_alpha, frame,
                      1.0 - config.prediction_overlap_alpha, 0.0, frame);
    } else {
      draw_dashed_quad(frame, state.future_target.corners, future_color, 2);
    }
    if (draw_labels) {
      const std::string label = "A" + std::to_string(state.future_target.id) + " FUT +" +
          std::to_string(state.prediction_lead_frames) + "F";
      cv::putText(frame, label, state.future_target.center + cv::Point2f(4, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.43, future_color, 1, cv::LINE_AA);
    }
  }

  const int panel_width = std::min(frame.cols - 20, 540);
  const int panel_top = frame.rows - 125;
  cv::rectangle(frame, {10, panel_top}, {10 + panel_width, frame.rows - 10},
                cv::Scalar(10, 10, 10), cv::FILLED);
  std::string detection_state = "NO DETECTION";
  if (state.candidate_available) {
    detection_state = "DET A" + std::to_string(state.detected_slot_index + 1) +
        (state.candidate_measurement_used ? " / USED" : " / GATED");
  }
  cv::putText(frame, "frame=" + std::to_string(frame_number) + "  " + detection_state,
              {20, panel_top + 24}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
              cv::Scalar(225, 225, 225), 1,
              cv::LINE_AA);
  std::ostringstream line2, line3, line4;
  line2 << "direction=" << state.direction << "  omega=" << std::fixed << std::setprecision(1)
        << std::abs(state.angular_speed_rad_s) * 180.0 / CV_PI << " deg/s";
  line3 << "mode=" << mode_name(state.mode) << "  future=+"
        << state.prediction_lead_frames << "F / +" << std::fixed << std::setprecision(3)
        << state.prediction_lead_s << "s  predictor="
        << (state.image_motion_prediction_used
                ? (state.image_prediction_coasting
                       ? "IMAGE-COAST"
                       : (state.future_target_filter_used ? "IMAGE+STATE" : "IMAGE"))
                : "PHASE")
        << "  missed=" << state.missed_frames;
  line4 << "green=current detection/model  red=future target"
        << (state.prediction_valid
                ? (prediction_render.high_overlap ? " (dim overlap)" : "")
                : " (expired)");
  cv::putText(frame, line2.str(), {20, panel_top + 52}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(0, 220, 255), 2, cv::LINE_AA);
  cv::putText(frame, line3.str(), {20, panel_top + 80}, cv::FONT_HERSHEY_SIMPLEX, 0.50,
              cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
  cv::putText(frame, line4.str(), {20, panel_top + 106}, cv::FONT_HERSHEY_SIMPLEX, 0.47,
              cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
  (void)debug_lights;
}

std::vector<ArmorObservation> collect_geometry_samples(const std::string& input_path,
                                                       const Config& config,
                                                       cv::Size* image_size,
                                                       double* fps_out) {
  cv::VideoCapture capture(input_path);
  if (!capture.isOpened()) throw std::runtime_error("cannot open input video: " + input_path);
  *image_size = {static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
                 static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT))};
  *fps_out = capture.get(cv::CAP_PROP_FPS);
  if (*fps_out < 1.0) *fps_out = 30.0;
  FramePreprocessor preprocessor(config, *image_size);
  ArmorDetector detector(config);
  std::vector<ArmorObservation> samples;
  cv::Mat raw;
  int frame_number = 0;
  while (capture.read(raw)) {
    if (frame_number % std::max(1, config.geometry_calibration_stride) == 0) {
      const DetectionFrame detections = detector.detect(preprocessor.process(raw));
      if (!detections.armors.empty()) samples.push_back(detections.armors.front());
    }
    ++frame_number;
  }
  return samples;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <input.mp4> [output.mp4] [config.yaml] [--no-gui] [--debug-lights]\n";
    return 1;
  }
  const std::string input_path = argv[1];
  const std::string output_path = argc >= 3 && argv[2][0] != '-' ? argv[2] : "output/result.mp4";
  const std::string config_path = argc >= 4 && argv[3][0] != '-' ? argv[3] : "config/video.yaml";
  bool show_gui = true, debug_lights = false;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--no-gui") show_gui = false;
    if (arg == "--debug-lights") debug_lights = true;
  }

  try {
    Config config = Config::load(config_path);
    if (config.solver_mode == "pnp")
      throw std::runtime_error("pnp mode requires matching calibration and is intentionally disabled; use affine or auto");
    cv::Size image_size;
    double fps = 0.0;
    const auto samples = collect_geometry_samples(input_path, config, &image_size, &fps);
    config.resolve_prediction_horizon(fps);
    std::cout << "Config: camera_undistort=" << (config.camera_enabled ? "on" : "off")
              << ", physical=(armor " << config.armor_width_m << 'x'
              << config.armor_height_m << " m, radius " << config.rotation_radius_m
              << " m, pitch " << config.outpost_pitch_deg << " deg)"
              << ", speed_snap=" << (config.angular_speed_snap_enabled ? "on" : "off")
              << " (" << config.angular_speed_snap_rad_s << " +/- "
              << config.angular_speed_snap_tolerance_rad_s << " rad/s)"
              << ", missed_frames=" << config.prediction_display_max_missed_frames
              << '/' << config.max_prediction_frames << '\n';
    const AffineGeometry geometry = calibrate_affine_geometry(samples, config, image_size);
    const PhaseQuadModel phase_quad_model = calibrate_phase_quad_model(samples, geometry, config);
    if (!phase_quad_model.valid())
      throw std::runtime_error("phase quadrilateral model calibration failed: insufficient phase coverage");
    std::cout << "Geometry: samples=" << geometry.calibration_samples
              << " center=(" << geometry.center.x << ',' << geometry.center.y << ')'
              << " axes=(" << cv::norm(geometry.axis_cos) << ',' << cv::norm(geometry.axis_sin) << ')'
              << " rms=" << geometry.calibration_rms_px << " px\n"
              << "Phase quad model: valid"
              << ", covered_bins=" << phase_quad_model.covered_bins() << '/'
              << phase_quad_model.bin_count() << '\n'
              << "Future prediction: +" << config.prediction_lead_frames
              << " frames / +" << config.prediction_lead_s << " s\n";
    const MotionPrior motion_prior = estimate_motion_prior(input_path, config, geometry, image_size, fps);
    if (motion_prior.valid) {
      std::cout << "Motion prior estimated from observations: repeat="
                << motion_prior.repeat_period_frames << " frames, speed="
                << motion_prior.speed_abs_rad_s << " rad/s, phase_sign="
                << motion_prior.phase_direction_sign << ", active_frames="
                << motion_prior.detected_motion_start_frame << ".."
                << motion_prior.detected_motion_end_frame << '\n';
      if (motion_prior.handover.valid) {
        std::cout << "Handover prior: first="
                  << motion_prior.handover.first_transition_frame
                  << ", direction_normalized=("
                  << motion_prior.handover.forward_direction_normalized.x << ','
                  << motion_prior.handover.forward_direction_normalized.y << ")\n";
        for (int slot = 0; slot < 3; ++slot) {
          const auto& learned = motion_prior.handover.slots[slot];
          std::cout << "  A" << slot + 1 << ": events=" << learned.transition_samples
                    << ", score=progress-" << learned.progress_age_weight_px
                    << "*age <= " << learned.progress_threshold
                    << ", age_gate=" << learned.age_threshold_frames << "\n";
        }
      }
    } else {
      std::cout << "Motion prior unavailable; falling back to online phase regression.\n";
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) throw std::runtime_error("cannot reopen input video");
    FramePreprocessor preprocessor(config, image_size);
    ArmorDetector detector(config);
    RigidArmorSolver solver(config, geometry, phase_quad_model, motion_prior);
    OnlineMotionDetector motion_detector(config);
    PredictionRenderState prediction_render_state;

    const auto output_parent = std::filesystem::path(output_path).parent_path();
    if (!output_parent.empty()) std::filesystem::create_directories(output_parent);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, image_size);
    if (!writer.isOpened()) throw std::runtime_error("cannot create output video: " + output_path);
    std::filesystem::path csv_path(output_path);
    csv_path.replace_extension(".csv");
    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("cannot create CSV: " + csv_path.string());
    csv << "frame,time_s,candidate_available,measurement_used,candidate_measurement_used,"
           "image_motion_prediction_used,"
           "image_prediction_coasting,image_candidate_used,image_handover_detected,"
           "image_future_handover,image_handover_spatial_vote,image_handover_progress_vote,"
           "image_handover_age_vote,image_handover_vote_count,image_track_age_frames,"
           "image_track_progress,image_normalized_step_px,detected_slot_index,"
           "measurement_slot_index,candidate_association_slot_index,"
           "candidate_x,candidate_y,raw_ellipse_phase_rad,phase_rad,future_phase_rad,"
           "prediction_lead_s,omega_rad_s,alpha_rad_s2,direction,mode,reprojection_error_px,"
           "missed_frames,future_target_valid,future_slot_index,future_target_id,"
           "future_target_x,future_target_y,"
           "raw_future_target_valid,raw_future_target_id,raw_future_target_x,"
           "raw_future_target_y,future_target_filter_used,future_target_filter_reset,"
           "future_target_filter_innovation_px,future_target_filter_velocity_x,"
           "future_target_filter_velocity_y,future_render_drawn,"
           "future_render_high_overlap,future_render_distance_px,future_render_iou,"
           "future_model1_x,future_model1_y,future_model1_area,"
           "future_model2_x,future_model2_y,future_model2_area,future_model3_x,future_model3_y,future_model3_area,"
           "slot1_source,slot1_x,slot1_y,slot2_source,slot2_x,slot2_y,"
           "slot3_source,slot3_x,slot3_y\n";

    cv::Mat raw;
    int frame_number = 0, measured_frames = 0;
    while (capture.read(raw)) {
      cv::Mat frame = preprocessor.process(raw);
      const DetectionFrame detections = detector.detect(frame);
      const bool scene_moving = motion_detector.update(frame);
      double timestamp_s = capture.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
      if (!(timestamp_s > 0.0)) timestamp_s = frame_number / fps;
      const SolverOutput state = solver.update(detections.armors, timestamp_s, scene_moving);
      if (state.measurement_used) ++measured_frames;
      const PredictionRenderDecision prediction_render =
          evaluate_prediction_render(state, config, &prediction_render_state);
      draw_overlay(frame, state, frame_number, config, prediction_render, true,
                   debug_lights);
      if (debug_lights) {
        for (const auto& light : detections.lights) {
          cv::Point2f p[4]; light.rect.points(p);
          for (int i = 0; i < 4; ++i)
            cv::line(frame, p[i], p[(i + 1) % 4], cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
      }
      const double raw_phase = state.candidate_available ? geometry.phase_of(state.candidate.center) : 0.0;
      const auto projected_area = [](const ArmorSlotOutput& slot) {
        const std::vector<cv::Point2f> corners(slot.corners.begin(), slot.corners.end());
        return std::abs(cv::contourArea(corners));
      };
      csv << frame_number << ',' << std::fixed << std::setprecision(6) << timestamp_s << ','
          << state.candidate_available << ',' << state.measurement_used << ','
          << state.candidate_measurement_used << ','
          << state.image_motion_prediction_used << ',' << state.image_prediction_coasting << ','
          << state.image_candidate_used << ',' << state.image_handover_detected << ','
          << state.image_future_handover << ','
          << state.image_handover_spatial_vote << ','
          << state.image_handover_progress_vote << ','
          << state.image_handover_age_vote << ','
          << state.image_handover_vote_count << ','
          << state.image_track_age_frames << ','
          << state.image_track_progress << ','
          << state.image_normalized_step_px << ','
          << state.detected_slot_index << ',' << state.measurement_slot_index << ','
          << state.candidate_association_slot_index << ','
          << (state.candidate_available ? state.candidate.center.x : 0.0F) << ','
          << (state.candidate_available ? state.candidate.center.y : 0.0F) << ',' << raw_phase << ','
          << state.phase_rad << ',' << state.future_phase_rad << ','
          << state.prediction_lead_s << ',' << state.angular_speed_rad_s << ','
          << state.angular_acceleration_rad_s2 << ',' << state.direction << ','
          << mode_name(state.mode) << ',' << state.reprojection_error_px << ','
          << state.missed_frames << ',' << state.future_target.valid << ','
          << state.future_slot_index << ','
          << state.future_target.id << ',' << state.future_target.center.x << ','
          << state.future_target.center.y << ','
          << state.raw_future_target.valid << ',' << state.raw_future_target.id << ','
          << state.raw_future_target.center.x << ','
          << state.raw_future_target.center.y << ','
          << state.future_target_filter_used << ','
          << state.future_target_filter_reset << ','
          << state.future_target_filter_innovation_px << ','
          << state.future_target_filter_velocity_px_s.x << ','
          << state.future_target_filter_velocity_px_s.y << ','
          << prediction_render.draw << ',' << prediction_render.high_overlap << ','
          << prediction_render.center_distance_px << ','
          << prediction_render.quad_iou << ','
          << state.future_model_slots[0].center.x << ','
          << state.future_model_slots[0].center.y << ',' << projected_area(state.future_model_slots[0]) << ','
          << state.future_model_slots[1].center.x << ','
          << state.future_model_slots[1].center.y << ',' << projected_area(state.future_model_slots[1]) << ','
          << state.future_model_slots[2].center.x << ','
          << state.future_model_slots[2].center.y << ',' << projected_area(state.future_model_slots[2]) << ','
          << source_name(state.slots[0].source) << ','
          << state.slots[0].center.x << ',' << state.slots[0].center.y << ','
          << source_name(state.slots[1].source) << ',' << state.slots[1].center.x << ','
          << state.slots[1].center.y << ',' << source_name(state.slots[2].source) << ','
          << state.slots[2].center.x << ',' << state.slots[2].center.y
          << '\n';
      writer.write(frame);
      if (show_gui) {
        cv::imshow("Outpost three-armor solver", frame);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 27 || key == 'q') break;
        if (key == ' ') while ((cv::waitKey(0) & 0xff) != ' ');
      }
      ++frame_number;
    }
    if (show_gui) cv::destroyAllWindows();
    std::cout << "Processed " << frame_number << " frames; measurements used=" << measured_frames
              << "\nVideo: " << output_path << "\nCSV: " << csv_path << '\n';
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 2;
  }
  return 0;
}
