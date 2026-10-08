#include "rigid_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace {
constexpr double kTwoPi = 2.0 * CV_PI;

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  double value = *middle;
  if (values.size() % 2 == 0) {
    const auto lower = std::max_element(values.begin(), middle);
    value = 0.5 * (value + *lower);
  }
  return value;
}

double wrap_positive(double phase) {
  phase = std::fmod(phase, kTwoPi);
  return phase < 0.0 ? phase + kTwoPi : phase;
}

cv::Point2d lerp(const cv::Point2d& first, const cv::Point2d& second, double amount) {
  return first * (1.0 - amount) + second * amount;
}

cv::Point2d limited_step(const cv::Point2d& step, double maximum_length) {
  const double length = cv::norm(step);
  if (length <= maximum_length || length < 1e-9) return step;
  return step * (maximum_length / length);
}

bool valid_projected_quad(const std::array<cv::Point2f, 4>& corners) {
  const std::vector<cv::Point2f> contour(corners.begin(), corners.end());
  return cv::isContourConvex(contour) && std::abs(cv::contourArea(contour)) >= 1.0;
}

AffineGeometry geometry_from_ellipse(const cv::RotatedRect& ellipse) {
  AffineGeometry geometry;
  geometry.center = ellipse.center;
  const double angle = ellipse.angle * CV_PI / 180.0;
  const double a = 0.5 * ellipse.size.width;
  const double b = 0.5 * ellipse.size.height;
  geometry.axis_cos = {a * std::cos(angle), a * std::sin(angle)};
  geometry.axis_sin = {-b * std::sin(angle), b * std::cos(angle)};
  return geometry;
}

double normalized_radius_error(const AffineGeometry& geometry, const cv::Point2d& point) {
  const double phase = geometry.phase_of(point);
  const cv::Point2d on_ellipse = geometry.point(phase);
  const double scale = 0.5 * (cv::norm(geometry.axis_cos) + cv::norm(geometry.axis_sin));
  return cv::norm(point - on_ellipse) / std::max(1.0, scale);
}

bool plausible(const cv::RotatedRect& ellipse, cv::Size image_size) {
  const double small = std::min(ellipse.size.width, ellipse.size.height);
  const double large = std::max(ellipse.size.width, ellipse.size.height);
  const double maximum = 1.25 * std::max(image_size.width, image_size.height);
  return small >= 10.0 && large >= 24.0 && large <= maximum &&
         ellipse.center.x > -0.25 * image_size.width &&
         ellipse.center.x < 1.25 * image_size.width &&
         ellipse.center.y > -0.25 * image_size.height &&
         ellipse.center.y < 1.25 * image_size.height;
}

}  // namespace

cv::Point2d AffineGeometry::point(double phase) const {
  return center + axis_cos * std::cos(phase) + axis_sin * std::sin(phase);
}

cv::Point2d AffineGeometry::tangent(double phase) const {
  return axis_cos * (-std::sin(phase)) + axis_sin * std::cos(phase);
}

double AffineGeometry::phase_of(const cv::Point2d& point_value) const {
  const cv::Point2d delta = point_value - center;
  const double determinant = axis_cos.x * axis_sin.y - axis_cos.y * axis_sin.x;
  if (std::abs(determinant) < 1e-9) throw std::runtime_error("degenerate ellipse geometry");
  const double cosine = (delta.x * axis_sin.y - delta.y * axis_sin.x) / determinant;
  const double sine = (axis_cos.x * delta.y - axis_cos.y * delta.x) / determinant;
  return std::atan2(sine, cosine);
}

AffineGeometry calibrate_affine_geometry(const std::vector<ArmorObservation>& samples,
                                         const Config& config, cv::Size image_size) {
  if (samples.size() < static_cast<std::size_t>(config.min_geometry_samples))
    throw std::runtime_error("not enough armor observations to calibrate projection geometry");
  std::vector<cv::Point2f> points;
  points.reserve(samples.size());
  for (const auto& sample : samples) points.push_back(sample.center);

  std::mt19937 random(0x524D2026U);
  std::uniform_int_distribution<std::size_t> choose(0, points.size() - 1);
  std::vector<int> best_inliers;
  double best_error = std::numeric_limits<double>::infinity();
  for (int iteration = 0; iteration < config.ellipse_ransac_iterations; ++iteration) {
    std::vector<cv::Point2f> subset;
    while (subset.size() < 5) {
      const cv::Point2f candidate = points[choose(random)];
      bool duplicate = false;
      for (const auto& existing : subset) if (cv::norm(candidate - existing) < 0.5) duplicate = true;
      if (!duplicate) subset.push_back(candidate);
    }
    cv::RotatedRect ellipse;
    try { ellipse = cv::fitEllipse(subset); } catch (const cv::Exception&) { continue; }
    if (!plausible(ellipse, image_size)) continue;
    const AffineGeometry candidate = geometry_from_ellipse(ellipse);
    std::vector<int> inliers;
    double error = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
      const double e = normalized_radius_error(candidate, points[i]);
      if (e <= config.ellipse_inlier_threshold) { inliers.push_back(static_cast<int>(i)); error += e; }
    }
    if (inliers.size() > best_inliers.size() ||
        (inliers.size() == best_inliers.size() && error < best_error)) {
      best_inliers = std::move(inliers);
      best_error = error;
    }
  }
  if (best_inliers.size() < static_cast<std::size_t>(config.min_geometry_samples)) {
    best_inliers.resize(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) best_inliers[i] = static_cast<int>(i);
  }
  std::vector<cv::Point2f> inlier_points;
  for (int index : best_inliers) inlier_points.push_back(points[index]);
  cv::RotatedRect fitted = cv::fitEllipse(inlier_points);
  if (!plausible(fitted, image_size)) throw std::runtime_error("ellipse calibration is degenerate");
  AffineGeometry geometry = geometry_from_ellipse(fitted);

  std::vector<double> width_scales, heights;
  double squared_error = 0.0;
  for (int index : best_inliers) {
    const auto& sample = samples[index];
    const double phase = geometry.phase_of(sample.center);
    const double tangent_length = cv::norm(geometry.tangent(phase));
    if (tangent_length > 1.0)
      width_scales.push_back(sample.size.width / (2.0 * tangent_length));
    heights.push_back(sample.size.height);
    const double pixel_error = cv::norm(cv::Point2d(sample.center) - geometry.point(phase));
    squared_error += pixel_error * pixel_error;
  }
  geometry.armor_width_scale = std::clamp(median(width_scales), 0.10, 0.80);
  geometry.armor_height_px = std::clamp(median(heights), 6.0, 80.0);
  geometry.calibration_samples = static_cast<int>(best_inliers.size());
  geometry.calibration_rms_px = std::sqrt(squared_error / std::max<std::size_t>(1, best_inliers.size()));
  return geometry;
}

PhaseQuadModel calibrate_phase_quad_model(const std::vector<ArmorObservation>& samples,
                                          const AffineGeometry& geometry,
                                          const Config& config) {
  PhaseQuadModel model;
  const int bin_count = std::max(1, config.shape_phase_bins);
  model.bins_.resize(static_cast<std::size_t>(bin_count));

  struct BinSamples {
    std::vector<double> center_x;
    std::vector<double> center_y;
    std::array<std::vector<double>, 4> corner_x;
    std::array<std::vector<double>, 4> corner_y;
  };
  std::vector<BinSamples> accumulated(static_cast<std::size_t>(bin_count));

  for (const auto& sample : samples) {
    double phase = 0.0;
    try {
      phase = wrap_positive(geometry.phase_of(sample.center));
    } catch (const std::runtime_error&) {
      continue;
    }
    const double bin_coordinate = phase * bin_count / kTwoPi;
    const int bin_index = static_cast<int>(std::floor(bin_coordinate + 0.5)) % bin_count;
    auto& values = accumulated[static_cast<std::size_t>(bin_index)];
    const cv::Point2d center_residual =
        cv::Point2d(sample.center) - geometry.point(phase);
    values.center_x.push_back(center_residual.x);
    values.center_y.push_back(center_residual.y);
    for (int corner = 0; corner < 4; ++corner) {
      const cv::Point2d offset =
          cv::Point2d(sample.corners[corner]) - cv::Point2d(sample.center);
      values.corner_x[corner].push_back(offset.x);
      values.corner_y[corner].push_back(offset.y);
    }
  }

  std::vector<bool> covered(static_cast<std::size_t>(bin_count), false);
  const int minimum_samples = std::max(1, config.shape_min_samples_per_bin);
  for (int bin = 0; bin < bin_count; ++bin) {
    const auto& values = accumulated[static_cast<std::size_t>(bin)];
    if (static_cast<int>(values.center_x.size()) < minimum_samples) continue;
    covered[static_cast<std::size_t>(bin)] = true;
    ++model.covered_bins_;
    auto& output = model.bins_[static_cast<std::size_t>(bin)];
    output.center_residual = {median(values.center_x), median(values.center_y)};
    for (int corner = 0; corner < 4; ++corner) {
      output.corner_offsets[corner] = {
          median(values.corner_x[corner]), median(values.corner_y[corner])};
    }
  }

  const int required_coverage = std::clamp(config.shape_min_covered_bins, 1, bin_count);
  if (model.covered_bins_ < required_coverage) return model;

  // Fill gaps by circular interpolation before smoothing, so the +/-pi seam is
  // treated exactly like every other pair of neighboring phase bins.
  const auto robust_bins = model.bins_;
  for (int bin = 0; bin < bin_count; ++bin) {
    if (covered[static_cast<std::size_t>(bin)]) continue;
    int previous_distance = 1;
    while (previous_distance < bin_count &&
           !covered[static_cast<std::size_t>(
               (bin - previous_distance + bin_count) % bin_count)])
      ++previous_distance;
    int next_distance = 1;
    while (next_distance < bin_count &&
           !covered[static_cast<std::size_t>((bin + next_distance) % bin_count)])
      ++next_distance;
    const int previous = (bin - previous_distance + bin_count) % bin_count;
    const int next = (bin + next_distance) % bin_count;
    const double amount = static_cast<double>(previous_distance) /
                          static_cast<double>(previous_distance + next_distance);
    auto& output = model.bins_[static_cast<std::size_t>(bin)];
    output.center_residual =
        lerp(robust_bins[static_cast<std::size_t>(previous)].center_residual,
             robust_bins[static_cast<std::size_t>(next)].center_residual, amount);
    for (int corner = 0; corner < 4; ++corner) {
      output.corner_offsets[corner] =
          lerp(robust_bins[static_cast<std::size_t>(previous)].corner_offsets[corner],
               robust_bins[static_cast<std::size_t>(next)].corner_offsets[corner], amount);
    }
  }

  const int smoothing_radius =
      std::clamp(config.shape_smoothing_radius_bins, 0, (bin_count - 1) / 2);
  if (smoothing_radius > 0) {
    const auto filled_bins = model.bins_;
    const double weight = 1.0 / (2.0 * smoothing_radius + 1.0);
    for (int bin = 0; bin < bin_count; ++bin) {
      auto& output = model.bins_[static_cast<std::size_t>(bin)];
      output.center_residual = {};
      output.corner_offsets = {};
      for (int offset = -smoothing_radius; offset <= smoothing_radius; ++offset) {
        const int source = (bin + offset + bin_count) % bin_count;
        output.center_residual +=
            filled_bins[static_cast<std::size_t>(source)].center_residual * weight;
        for (int corner = 0; corner < 4; ++corner) {
          output.corner_offsets[corner] +=
              filled_bins[static_cast<std::size_t>(source)].corner_offsets[corner] * weight;
        }
      }
    }
  }
  model.valid_ = true;
  return model;
}

ProjectedArmor PhaseQuadModel::project(int id, double phase,
                                       const AffineGeometry& geometry,
                                       const cv::Point2d& model_offset) const {
  ProjectedArmor output;
  output.id = id;
  if (!valid_ || bins_.empty()) return output;
  output.valid = true;

  const double bin_coordinate = wrap_positive(phase) * bins_.size() / kTwoPi;
  const int first_index = static_cast<int>(std::floor(bin_coordinate)) %
                          static_cast<int>(bins_.size());
  const int second_index = (first_index + 1) % static_cast<int>(bins_.size());
  const double amount = bin_coordinate - std::floor(bin_coordinate);
  const auto& first = bins_[static_cast<std::size_t>(first_index)];
  const auto& second = bins_[static_cast<std::size_t>(second_index)];
  const cv::Point2d center_residual =
      lerp(first.center_residual, second.center_residual, amount);
  const cv::Point2d center = geometry.point(phase) + center_residual + model_offset;
  output.center = cv::Point2f(center);
  for (int corner = 0; corner < 4; ++corner) {
    const cv::Point2d offset =
        lerp(first.corner_offsets[corner], second.corner_offsets[corner], amount);
    output.corners[corner] = cv::Point2f(center + offset);
  }
  if (!valid_projected_quad(output.corners)) {
    const auto& nearest = amount < 0.5 ? first : second;
    for (int corner = 0; corner < 4; ++corner)
      output.corners[corner] = cv::Point2f(center + nearest.corner_offsets[corner]);
    output.valid = valid_projected_quad(output.corners);
  }
  return output;
}

RigidArmorSolver::RigidArmorSolver(const Config& config, AffineGeometry geometry,
                                   PhaseQuadModel phase_quad_model,
                                   MotionPrior motion_prior)
    : config_(config),
      geometry_(std::move(geometry)),
      phase_quad_model_(std::move(phase_quad_model)),
      motion_prior_(motion_prior) {
  if (!phase_quad_model_.valid())
    throw std::runtime_error("phase quadrilateral model is invalid; refusing rectangle fallback");
}

std::array<ProjectedArmor, 3> RigidArmorSolver::project_all(double base_phase) const {
  std::array<ProjectedArmor, 3> projected;
  for (int i = 0; i < 3; ++i) {
    const double phase = base_phase + i * kTwoPi / 3.0;
    projected[i] = phase_quad_model_.project(i + 1, phase, geometry_, model_offset_);
  }
  return projected;
}

MotionMode RigidArmorSolver::classify_mode(double dt) {
  MotionMode candidate;
  const double speed_deg = std::abs(speed_) * 180.0 / CV_PI;
  const double acceleration_deg = std::abs(acceleration_) * 180.0 / CV_PI;
  if (missed_frames_ > config_.max_prediction_frames) candidate = MotionMode::kLost;
  else if (!scene_moving_ && speed_deg < config_.stationary_speed_deg_s) candidate = MotionMode::kStopped;
  else if (previous_speed_ * speed_ < 0.0 &&
           std::abs(previous_speed_) * 180.0 / CV_PI > config_.stationary_speed_deg_s)
    candidate = MotionMode::kReversing;
  else if (acceleration_deg < config_.uniform_acceleration_deg_s2) candidate = MotionMode::kUniform;
  else if (acceleration_ * speed_ > 0.0) candidate = MotionMode::kAccelerating;
  else candidate = MotionMode::kDecelerating;

  if (candidate == pending_mode_) ++pending_mode_frames_;
  else { pending_mode_ = candidate; pending_mode_frames_ = 1; }
  if (pending_mode_frames_ >= config_.mode_hold_frames || committed_mode_ == MotionMode::kInitializing)
    committed_mode_ = candidate;
  (void)dt;
  return committed_mode_;
}

double RigidArmorSolver::robust_phase_slope() const {
  if (phase_samples_.size() < 5) return speed_;
  auto fit = [&](const std::vector<bool>* keep) {
    double sw = 0.0, st = 0.0, sp = 0.0, stt = 0.0, stp = 0.0;
    const double origin = phase_samples_.front().time;
    for (std::size_t i = 0; i < phase_samples_.size(); ++i) {
      if (keep && !(*keep)[i]) continue;
      const double t = phase_samples_[i].time - origin;
      const double p = phase_samples_[i].phase;
      sw += 1.0; st += t; sp += p; stt += t * t; stp += t * p;
    }
    const double denominator = sw * stt - st * st;
    return std::abs(denominator) < 1e-9 ? speed_ : (sw * stp - st * sp) / denominator;
  };
  double slope = fit(nullptr);
  const double origin = phase_samples_.front().time;
  double mean_t = 0.0, mean_p = 0.0;
  for (const auto& sample : phase_samples_) {
    mean_t += sample.time - origin;
    mean_p += sample.phase;
  }
  mean_t /= phase_samples_.size(); mean_p /= phase_samples_.size();
  const double intercept = mean_p - slope * mean_t;
  std::vector<double> residuals;
  for (const auto& sample : phase_samples_)
    residuals.push_back(std::abs(sample.phase - (intercept + slope * (sample.time - origin))));
  const double mad = median(residuals);
  std::vector<bool> keep(residuals.size(), true);
  const double cutoff = std::max(0.035, 3.0 * 1.4826 * mad);
  for (std::size_t i = 0; i < residuals.size(); ++i) keep[i] = residuals[i] <= cutoff;
  return fit(&keep);
}

SolverOutput RigidArmorSolver::update(const std::vector<ArmorObservation>& observations,
                                      double timestamp_s, bool scene_moving) {
  SolverOutput output;
  scene_moving_ = scene_moving;
  double dt = initialized_ ? timestamp_s - last_timestamp_s_ : 1.0 / 30.0;
  dt = std::clamp(dt, 1e-3, 0.2);
  const double prediction_lead_s = config_.prediction_lead_s > 0.0
      ? config_.prediction_lead_s
      : config_.prediction_lead_frames / 30.0;
  const double predicted_phase = phase_ + speed_ * dt;

  int best_observation = -1, best_slot = -1;
  double best_candidate_phase = predicted_phase;
  double best_pixel_error = std::numeric_limits<double>::infinity();
  double best_cost = std::numeric_limits<double>::infinity();
  for (std::size_t observation_index = 0; observation_index < observations.size(); ++observation_index) {
    const auto& observation = observations[observation_index];
    const double measured_ellipse_phase = geometry_.phase_of(observation.center);
    for (int slot = 0; slot < 3; ++slot) {
      const double raw_base = measured_ellipse_phase - slot * kTwoPi / 3.0;
      const double candidate_phase = predicted_phase +
          std::remainder(raw_base - predicted_phase, kTwoPi);
      const cv::Point2d association_center =
          geometry_.point(predicted_phase + slot * kTwoPi / 3.0);
      const double pixel_error = cv::norm(
          association_center - cv::Point2d(observation.center));
      const double phase_error = std::abs(candidate_phase - predicted_phase);
      const double cost = pixel_error + 12.0 * phase_error - 3.0 * observation.score;
      if (cost < best_cost) {
        best_cost = cost; best_observation = static_cast<int>(observation_index);
        best_slot = slot; best_candidate_phase = candidate_phase;
        best_pixel_error = pixel_error;
      }
    }
  }
  int candidate_observation = best_observation;
  int accepted_slot = -1;
  bool accepted_measurement = false;
  const double innovation_limit = config_.max_phase_innovation_deg * CV_PI / 180.0;
  const bool regular_match = initialized_ && best_observation >= 0 &&
      best_pixel_error <= config_.max_observation_distance_px &&
      std::abs(best_candidate_phase - predicted_phase) <= innovation_limit;
  // Once the scene has stopped, the last moving prediction can be displaced by
  // braking motion that happened inside the motion-detector hold window. Reacquire
  // a real observation with a wider one-off gate instead of reporting a false loss.
  const bool stationary_reacquire = initialized_ && best_observation >= 0 && !scene_moving_ &&
      best_pixel_error <= 2.5 * config_.max_observation_distance_px &&
      std::abs(best_candidate_phase - predicted_phase) <= 85.0 * CV_PI / 180.0;

  if (!initialized_ && !observations.empty()) {
    const auto best = std::max_element(observations.begin(), observations.end(),
                                      [](const auto& a, const auto& b) { return a.score < b.score; });
    phase_ = geometry_.phase_of(best->center);
    speed_ = 0.0; acceleration_ = 0.0; previous_speed_ = 0.0;
    initialized_ = true; missed_frames_ = 0; best_observation = static_cast<int>(best - observations.begin());
    best_slot = 0; best_candidate_phase = phase_;
    candidate_observation = best_observation;
    accepted_slot = best_slot;
    accepted_measurement = true;
    phase_samples_.push_back({timestamp_s, phase_});
  } else if (regular_match || stationary_reacquire) {
    const double residual = best_candidate_phase - predicted_phase;
    previous_speed_ = speed_;
    const double correction_gain = stationary_reacquire ? 0.85 : config_.phase_gain;
    phase_ = predicted_phase + correction_gain * residual;
    if (scene_moving_) {
      phase_samples_.push_back({timestamp_s, best_candidate_phase});
      while (!phase_samples_.empty() &&
             timestamp_s - phase_samples_.front().time > config_.phase_regression_window_s)
        phase_samples_.pop_front();
      const double measured_speed = robust_phase_slope();
      bool accept_measured_speed = true;
      if (motion_prior_.valid) {
        const double signed_prior = motion_prior_.phase_direction_sign * motion_prior_.speed_abs_rad_s;
        accept_measured_speed = measured_speed * signed_prior > 0.0 &&
            std::abs(measured_speed) >
                config_.speed_measurement_min_ratio * motion_prior_.speed_abs_rad_s &&
            std::abs(measured_speed) <
                config_.speed_measurement_max_ratio * motion_prior_.speed_abs_rad_s;
        if (std::abs(speed_) < 0.20) speed_ = signed_prior;
        if (accept_measured_speed)
          speed_ = (1.0 - config_.speed_gain) * speed_ + config_.speed_gain * measured_speed;
        speed_ = (1.0 - config_.speed_prior_gain) * speed_ +
                 config_.speed_prior_gain * signed_prior;
      } else if (accept_measured_speed) {
        speed_ = (1.0 - config_.speed_gain) * speed_ + config_.speed_gain * measured_speed;
      }
    } else {
      phase_samples_.clear();
    }
    const double max_speed = config_.max_abs_speed_deg_s * CV_PI / 180.0;
    speed_ = std::clamp(speed_, -max_speed, max_speed);
    const double instantaneous_acceleration = (speed_ - previous_speed_) / dt;
    acceleration_ = (1.0 - config_.acceleration_gain) * acceleration_ +
                    config_.acceleration_gain * instantaneous_acceleration;
    missed_frames_ = 0;
    accepted_slot = best_slot;
    accepted_measurement = true;
  } else if (initialized_) {
    previous_speed_ = speed_;
    phase_ = predicted_phase;
    acceleration_ *= 0.92;
    ++missed_frames_;
  }
  if (initialized_ && !scene_moving_) {
    speed_ *= 0.68;
    if (std::abs(speed_) * 180.0 / CV_PI < 0.35 * config_.stationary_speed_deg_s) speed_ = 0.0;
  } else if (initialized_ && scene_moving_ && motion_prior_.valid && std::abs(speed_) < 0.20) {
    speed_ = motion_prior_.phase_direction_sign * motion_prior_.speed_abs_rad_s;
  }
  last_timestamp_s_ = timestamp_s;

  // A global image offset may compensate a small static camera/model mismatch,
  // but must never chase the armor tangentially while it rotates.  Otherwise it
  // hides a phase error in the current frame and makes future projection lag.
  if (accepted_measurement && candidate_observation >= 0 && accepted_slot >= 0 &&
      !scene_moving_) {
    const auto projected_before_offset = project_all(phase_);
    const cv::Point2d center_error =
        cv::Point2d(observations[static_cast<std::size_t>(candidate_observation)].center) -
        cv::Point2d(projected_before_offset[static_cast<std::size_t>(accepted_slot)].center);
    const cv::Point2d offset_step = limited_step(
        center_error * config_.model_offset_gain, config_.model_offset_max_step_px);
    model_offset_ += offset_step;
    model_offset_ = limited_step(model_offset_, config_.model_offset_max_magnitude_px);
  }

  output.candidate_available = candidate_observation >= 0;
  output.measurement_used = accepted_measurement;
  output.observed_slot = accepted_measurement ? accepted_slot : -1;
  output.missed_frames = missed_frames_;
  output.phase_rad = phase_;
  output.angular_speed_rad_s = speed_;
  output.angular_acceleration_rad_s2 = acceleration_;
  if (candidate_observation >= 0) {
    output.candidate = observations[static_cast<std::size_t>(candidate_observation)];
  }

  // Predict the next image position of the actually detected light-band pair.
  // This short-horizon predictor is independent from the phase-assimilation
  // gate: a rejected but geometrically valid detection still supplies current
  // image motion.  Velocity is a Theil-Sen robust line fit over the most recent
  // valid centers, rather than a noisy two-frame difference.  Large periodic
  // left-to-right jumps reset that window and are learned as transitions.
  std::optional<ArmorObservation> image_motion_future;
  if (candidate_observation >= 0) {
    const ArmorObservation& current =
        observations[static_cast<std::size_t>(candidate_observation)];
    bool periodic_jump = false;
    if (!detection_history_valid_ ||
        timestamp_s - previous_detection_time_s_ > 0.5) {
      display_candidate_slot_ = accepted_slot >= 0 ? accepted_slot : 0;
      detection_velocity_px_s_ = {};
      detection_velocity_valid_ = false;
      detection_history_.clear();
    } else {
      const double elapsed = std::max(1e-3, timestamp_s - previous_detection_time_s_);
      const cv::Point2d displacement =
          cv::Point2d(current.center) - cv::Point2d(previous_detection_.center);
      const double nominal_frame_s = prediction_lead_s /
          std::max(1, config_.prediction_lead_frames);
      const double normalized_step = cv::norm(displacement) * nominal_frame_s / elapsed;
      if (normalized_step >= config_.image_prediction_jump_min_px) {
        periodic_jump = true;
        jump_transitions_.push_back({previous_detection_, current});
        while (jump_transitions_.size() > 12) jump_transitions_.pop_front();
        display_candidate_slot_ = (std::max(0, display_candidate_slot_) + 1) % 3;
        detection_history_.clear();
      } else if (normalized_step > config_.image_prediction_max_step_px) {
        // An ambiguous step is neither a valid continuation nor a learned
        // periodic handover.  Start a fresh robust window without turning the
        // displacement into velocity.
        detection_history_.clear();
      }
    }

    detection_history_.push_back({timestamp_s, cv::Point2d(current.center)});
    while (detection_history_.size() >
           static_cast<std::size_t>(config_.image_prediction_window_frames))
      detection_history_.pop_front();
    if (!periodic_jump && detection_history_.size() >=
                              static_cast<std::size_t>(config_.image_prediction_min_samples)) {
      std::vector<double> slopes_x;
      std::vector<double> slopes_y;
      for (std::size_t first = 0; first < detection_history_.size(); ++first) {
        for (std::size_t second = first + 1; second < detection_history_.size(); ++second) {
          const double sample_dt = detection_history_[second].time_s -
                                   detection_history_[first].time_s;
          if (sample_dt <= 1e-4) continue;
          const cv::Point2d delta = detection_history_[second].center -
                                    detection_history_[first].center;
          slopes_x.push_back(delta.x / sample_dt);
          slopes_y.push_back(delta.y / sample_dt);
        }
      }
      if (!slopes_x.empty()) {
        const cv::Point2d robust_velocity(median(slopes_x), median(slopes_y));
        detection_velocity_px_s_ =
            detection_velocity_px_s_ * (1.0 - config_.image_prediction_velocity_gain) +
            robust_velocity * config_.image_prediction_velocity_gain;
        detection_velocity_valid_ = true;
      }
    }
    if (!scene_moving_) {
      detection_velocity_px_s_ *= 0.55;
      if (cv::norm(detection_velocity_px_s_) < 0.5) detection_velocity_px_s_ = {};
    }
    if (display_candidate_slot_ < 0)
      display_candidate_slot_ = accepted_slot >= 0 ? accepted_slot : std::max(0, best_slot);

    ArmorObservation predicted = current;
    cv::Point2d predicted_center = cv::Point2d(current.center) +
        detection_velocity_px_s_ * prediction_lead_s;
    cv::Point2d translation = predicted_center - cv::Point2d(current.center);
    predicted.center = cv::Point2f(predicted_center);
    for (auto& corner : predicted.corners) corner += cv::Point2f(translation);

    const double velocity_length = cv::norm(detection_velocity_px_s_);
    if (!periodic_jump && velocity_length > 1.0 && !jump_transitions_.empty()) {
      const cv::Point2d direction = detection_velocity_px_s_ * (1.0 / velocity_length);
      const auto nearest = std::min_element(
          jump_transitions_.begin(), jump_transitions_.end(),
          [&](const JumpTransition& first, const JumpTransition& second) {
            return std::abs(first.exit.center.y - current.center.y) <
                   std::abs(second.exit.center.y - current.center.y);
          });
      if (nearest != jump_transitions_.end() &&
          std::abs(nearest->exit.center.y - current.center.y) <=
              config_.image_prediction_transition_y_tolerance_px) {
        const double distance_to_exit =
            (cv::Point2d(nearest->exit.center) - cv::Point2d(current.center)).dot(direction);
        const double forecast_distance = velocity_length * prediction_lead_s;
        if (forecast_distance > distance_to_exit +
                                    config_.image_prediction_wrap_margin_px) {
          const double distance_after_entry = std::max(
              0.0, forecast_distance - std::max(0.0, distance_to_exit));
          predicted = nearest->entry;
          const cv::Point2d wrap_translation = direction * distance_after_entry;
          predicted.center += cv::Point2f(wrap_translation);
          for (auto& corner : predicted.corners) corner += cv::Point2f(wrap_translation);
        }
      }
    }
    if (detection_velocity_valid_) image_motion_future = predicted;
    previous_detection_ = current;
    previous_detection_time_s_ = timestamp_s;
    detection_history_valid_ = true;
  }
  output.candidate_slot = candidate_observation >= 0 ? display_candidate_slot_ : -1;
  const auto projected = project_all(phase_);
  if (accepted_measurement && accepted_slot >= 0) {
    output.reprojection_error_px = cv::norm(
        cv::Point2d(output.candidate.center) -
        cv::Point2d(projected[static_cast<std::size_t>(accepted_slot)].center));
  }
  const double orientation = geometry_.axis_cos.x * geometry_.axis_sin.y -
                             geometry_.axis_cos.y * geometry_.axis_sin.x;
  const double image_clockwise_speed = speed_ * (orientation >= 0.0 ? 1.0 : -1.0);
  if (std::abs(speed_) * 180.0 / CV_PI < config_.stationary_speed_deg_s) output.direction = "STOP";
  else output.direction = image_clockwise_speed > 0.0 ? "CW" : "CCW";
  if (accepted_measurement && committed_mode_ == MotionMode::kLost) {
    committed_mode_ = MotionMode::kInitializing;
    pending_mode_ = MotionMode::kInitializing;
    pending_mode_frames_ = 0;
  }
  output.mode = initialized_ ? classify_mode(dt) : MotionMode::kInitializing;
  output.model_valid = initialized_ &&
                       missed_frames_ <= config_.max_prediction_frames;
  output.prediction_lead_frames = config_.prediction_lead_frames;
  output.prediction_lead_s = prediction_lead_s;
  double prediction_acceleration = 0.0;
  if (missed_frames_ == 0 && scene_moving_ &&
      (output.mode == MotionMode::kAccelerating ||
       output.mode == MotionMode::kDecelerating)) {
    const double acceleration_limit =
        config_.prediction_max_acceleration_deg_s2 * CV_PI / 180.0;
    prediction_acceleration = std::clamp(
        acceleration_, -acceleration_limit, acceleration_limit);
  }
  output.future_phase_rad = phase_ + speed_ * output.prediction_lead_s +
      0.5 * prediction_acceleration * output.prediction_lead_s *
          output.prediction_lead_s;
  const bool current_model_visible = output.model_valid &&
      missed_frames_ <= config_.prediction_display_max_missed_frames;
  for (int slot = 0; slot < 3; ++slot)
    output.slots[static_cast<std::size_t>(slot)].id = slot + 1;
  if (current_model_visible) {
    for (int slot = 0; slot < 3; ++slot) {
      auto& destination = output.slots[static_cast<std::size_t>(slot)];
      destination.valid = projected[static_cast<std::size_t>(slot)].valid;
      destination.source = destination.valid ? BoxSource::kPredicted : BoxSource::kNone;
      destination.center = projected[static_cast<std::size_t>(slot)].center;
      destination.corners = projected[static_cast<std::size_t>(slot)].corners;
    }
  }
  // The green recognition box is always the exact current-frame detector quad,
  // regardless of whether the motion filter accepted the measurement.
  if (output.candidate_available && output.candidate_slot >= 0 &&
      output.candidate_slot < 3) {
    auto& destination = output.slots[static_cast<std::size_t>(output.candidate_slot)];
    destination.valid = true;
    destination.source = BoxSource::kObserved;
    destination.center = output.candidate.center;
    destination.corners = output.candidate.corners;
  }

  const int target_slot = output.candidate_slot >= 0
      ? output.candidate_slot : display_candidate_slot_;
  output.future_target.id = target_slot + 1;
  if (image_motion_future.has_value() && target_slot >= 0 && target_slot < 3) {
    output.future_target.valid = true;
    output.future_target.source = BoxSource::kPredicted;
    output.future_target.center = image_motion_future->center;
    output.future_target.corners = image_motion_future->corners;
    output.image_motion_prediction_used = true;
  } else if (current_model_visible && target_slot >= 0 && target_slot < 3) {
    const auto future_projected = project_all(output.future_phase_rad);
    const auto& target = future_projected[static_cast<std::size_t>(target_slot)];
    output.future_target.valid = target.valid;
    output.future_target.source = target.valid ? BoxSource::kPredicted : BoxSource::kNone;
    output.future_target.center = target.center;
    output.future_target.corners = target.corners;
  }
  output.prediction_valid = output.future_target.valid;
  return output;
}
