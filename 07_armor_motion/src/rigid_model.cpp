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

double snap_angular_speed(double speed, const Config& config) {
  if (!config.angular_speed_snap_enabled || std::abs(speed) < 1e-9) return speed;
  if (std::abs(std::abs(speed) - config.angular_speed_snap_rad_s) >
      config.angular_speed_snap_tolerance_rad_s)
    return speed;
  return std::copysign(config.angular_speed_snap_rad_s, speed);
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

double required_finite_double(const cv::FileNode& parent,
                              const std::string& name) {
  const cv::FileNode node = parent[name];
  if (node.empty())
    throw std::runtime_error("calibration profile is missing " + name);
  double value = 0.0;
  node >> value;
  if (!std::isfinite(value))
    throw std::runtime_error("calibration profile has non-finite " + name);
  return value;
}

int required_int(const cv::FileNode& parent, const std::string& name) {
  const cv::FileNode node = parent[name];
  if (node.empty())
    throw std::runtime_error("calibration profile is missing " + name);
  int value = 0;
  node >> value;
  return value;
}

bool valid_geometry(const AffineGeometry& geometry) {
  const auto finite_point = [](const cv::Point2d& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
  };
  const double determinant = geometry.axis_cos.x * geometry.axis_sin.y -
                             geometry.axis_cos.y * geometry.axis_sin.x;
  return finite_point(geometry.center) && finite_point(geometry.axis_cos) &&
         finite_point(geometry.axis_sin) &&
         cv::norm(geometry.axis_cos) > 1e-6 &&
         cv::norm(geometry.axis_sin) > 1e-6 &&
         std::abs(determinant) > 1e-6 &&
         std::isfinite(geometry.armor_width_scale) &&
         geometry.armor_width_scale > 0.0 &&
         std::isfinite(geometry.armor_height_px) &&
         geometry.armor_height_px > 0.0 &&
         geometry.calibration_samples >= 5 &&
         std::isfinite(geometry.calibration_rms_px) &&
         geometry.calibration_rms_px >= 0.0;
}

}  // namespace

double angular_speed_from_handover_interval(std::size_t handover_count,
                                            double interval_s) {
  if (handover_count == 0 || !std::isfinite(interval_s) || interval_s <= 0.0)
    throw std::invalid_argument("handover interval must be finite and positive");
  return static_cast<double>(handover_count) * (2.0 * CV_PI / 3.0) /
         interval_s;
}

cv::Point2d AffineGeometry::point(double phase) const {
  return center + axis_cos * std::cos(phase) + axis_sin * std::sin(phase);
}

cv::Point2d AffineGeometry::tangent(double phase) const {
  return axis_cos * (-std::sin(phase)) + axis_sin * std::cos(phase);
}

cv::Point2d AffineGeometry::normalized_coordinates(
    const cv::Point2d& point_value) const {
  const cv::Point2d delta = point_value - center;
  const double determinant = axis_cos.x * axis_sin.y - axis_cos.y * axis_sin.x;
  if (std::abs(determinant) < 1e-9) throw std::runtime_error("degenerate ellipse geometry");
  const double cosine = (delta.x * axis_sin.y - delta.y * axis_sin.x) / determinant;
  const double sine = (axis_cos.x * delta.y - axis_cos.y * delta.x) / determinant;
  return {cosine, sine};
}

double AffineGeometry::phase_of(const cv::Point2d& point_value) const {
  const cv::Point2d normalized = normalized_coordinates(point_value);
  return std::atan2(normalized.y, normalized.x);
}

AffineGeometry calibrate_affine_geometry(const std::vector<ArmorObservation>& samples,
                                         const Config& config, cv::Size image_size) {
  if (samples.size() < static_cast<std::size_t>(config.min_geometry_samples))
    throw std::runtime_error("not enough armor observations to calibrate projection geometry");
  std::vector<cv::Point2f> points;
  points.reserve(samples.size());
  for (const auto& sample : samples) points.push_back(sample.center);
  std::vector<cv::Point2f> unique_points;
  unique_points.reserve(points.size());
  for (const auto& point : points) {
    const bool duplicate = std::any_of(
        unique_points.begin(), unique_points.end(),
        [&](const cv::Point2f& existing) {
          return cv::norm(point - existing) < 0.5;
        });
    if (!duplicate) unique_points.push_back(point);
  }
  if (unique_points.size() < 5)
    throw std::runtime_error(
        "projection calibration needs at least five distinct armor centers");

  std::mt19937 random(0x524D2026U);
  std::vector<std::size_t> sample_indices(unique_points.size());
  for (std::size_t i = 0; i < sample_indices.size(); ++i) sample_indices[i] = i;
  std::vector<int> best_inliers;
  double best_error = std::numeric_limits<double>::infinity();
  for (int iteration = 0; iteration < config.ellipse_ransac_iterations; ++iteration) {
    std::shuffle(sample_indices.begin(), sample_indices.end(), random);
    std::vector<cv::Point2f> subset;
    subset.reserve(5);
    for (std::size_t i = 0; i < 5; ++i)
      subset.push_back(unique_points[sample_indices[i]]);
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

AffineGeometry calibrate_labeled_affine_geometry(
    const std::vector<PhaseLabeledObservation>& samples,
    const Config& config, cv::Size image_size) {
  if (samples.size() < static_cast<std::size_t>(config.min_geometry_samples))
    throw std::runtime_error("not enough phase-labeled observations to calibrate geometry");

  const auto fit = [&](const std::vector<int>& indices) {
    cv::Mat design(static_cast<int>(indices.size()), 3, CV_64F);
    cv::Mat x(static_cast<int>(indices.size()), 1, CV_64F);
    cv::Mat y(static_cast<int>(indices.size()), 1, CV_64F);
    for (int row = 0; row < static_cast<int>(indices.size()); ++row) {
      const auto& sample = samples[static_cast<std::size_t>(indices[row])];
      design.at<double>(row, 0) = 1.0;
      design.at<double>(row, 1) = std::cos(sample.phase_rad);
      design.at<double>(row, 2) = std::sin(sample.phase_rad);
      x.at<double>(row, 0) = sample.observation.center.x;
      y.at<double>(row, 0) = sample.observation.center.y;
    }
    cv::Mat coefficients_x, coefficients_y;
    if (!cv::solve(design, x, coefficients_x, cv::DECOMP_SVD) ||
        !cv::solve(design, y, coefficients_y, cv::DECOMP_SVD))
      throw std::runtime_error("phase-labeled affine calibration failed");
    AffineGeometry geometry;
    geometry.center = {coefficients_x.at<double>(0), coefficients_y.at<double>(0)};
    geometry.axis_cos = {coefficients_x.at<double>(1), coefficients_y.at<double>(1)};
    geometry.axis_sin = {coefficients_x.at<double>(2), coefficients_y.at<double>(2)};
    return geometry;
  };

  std::vector<int> indices(samples.size());
  for (std::size_t index = 0; index < samples.size(); ++index)
    indices[index] = static_cast<int>(index);
  AffineGeometry geometry = fit(indices);
  const double initial_scale = 0.5 *
      (cv::norm(geometry.axis_cos) + cv::norm(geometry.axis_sin));
  const double inlier_threshold = std::max(
      2.0, config.ellipse_inlier_threshold * std::max(1.0, initial_scale));
  std::vector<int> inliers;
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const auto& sample = samples[index];
    if (cv::norm(cv::Point2d(sample.observation.center) -
                 geometry.point(sample.phase_rad)) <= inlier_threshold)
      inliers.push_back(static_cast<int>(index));
  }
  if (inliers.size() >= static_cast<std::size_t>(config.min_geometry_samples))
    geometry = fit(inliers);
  else
    inliers = indices;

  const double determinant = geometry.axis_cos.x * geometry.axis_sin.y -
                             geometry.axis_cos.y * geometry.axis_sin.x;
  if (!std::isfinite(determinant) || std::abs(determinant) < 1e-6 ||
      cv::norm(geometry.axis_cos) < 5.0 || cv::norm(geometry.axis_sin) < 5.0 ||
      geometry.center.x < -0.25 * image_size.width ||
      geometry.center.x > 1.25 * image_size.width ||
      geometry.center.y < -0.25 * image_size.height ||
      geometry.center.y > 1.25 * image_size.height)
    throw std::runtime_error(
        "phase-labeled affine geometry is degenerate: determinant=" +
        std::to_string(determinant) + ", center=(" +
        std::to_string(geometry.center.x) + "," +
        std::to_string(geometry.center.y) + "), axes=(" +
        std::to_string(cv::norm(geometry.axis_cos)) + "," +
        std::to_string(cv::norm(geometry.axis_sin)) + ")");

  std::vector<double> width_scales, heights;
  double squared_error = 0.0;
  for (int index : inliers) {
    const auto& sample = samples[static_cast<std::size_t>(index)];
    const double tangent_length = cv::norm(geometry.tangent(sample.phase_rad));
    if (tangent_length > 1.0)
      width_scales.push_back(sample.observation.size.width / (2.0 * tangent_length));
    heights.push_back(sample.observation.size.height);
    const double error = cv::norm(cv::Point2d(sample.observation.center) -
                                  geometry.point(sample.phase_rad));
    squared_error += error * error;
  }
  geometry.armor_width_scale = std::clamp(median(width_scales), 0.10, 0.80);
  geometry.armor_height_px = std::clamp(median(heights), 6.0, 80.0);
  geometry.calibration_samples = static_cast<int>(inliers.size());
  geometry.calibration_rms_px = std::sqrt(
      squared_error / std::max<std::size_t>(1, inliers.size()));
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

PhaseQuadModel calibrate_labeled_phase_quad_model(
    const std::vector<PhaseLabeledObservation>& samples,
    const AffineGeometry& geometry, const Config& config) {
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
  for (const auto& labeled : samples) {
    const double phase = wrap_positive(labeled.phase_rad);
    const double bin_coordinate = phase * bin_count / kTwoPi;
    const int bin_index = static_cast<int>(std::floor(bin_coordinate + 0.5)) % bin_count;
    auto& values = accumulated[static_cast<std::size_t>(bin_index)];
    const auto& sample = labeled.observation;
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
  const int required_coverage = std::clamp(
      config.shape_min_covered_bins, 1, bin_count);
  if (model.covered_bins_ < required_coverage) return model;
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
    output.center_residual = lerp(
        robust_bins[static_cast<std::size_t>(previous)].center_residual,
        robust_bins[static_cast<std::size_t>(next)].center_residual, amount);
    for (int corner = 0; corner < 4; ++corner) {
      output.corner_offsets[corner] = lerp(
          robust_bins[static_cast<std::size_t>(previous)].corner_offsets[corner],
          robust_bins[static_cast<std::size_t>(next)].corner_offsets[corner], amount);
    }
  }
  const int smoothing_radius = std::clamp(
      config.shape_smoothing_radius_bins, 0, (bin_count - 1) / 2);
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

void PhaseQuadModel::write(cv::FileStorage& storage) const {
  storage << "valid" << static_cast<int>(valid_);
  storage << "covered_bins" << covered_bins_;
  storage << "bins" << "[";
  for (const auto& bin : bins_) {
    storage << "{";
    storage << "center_x" << bin.center_residual.x;
    storage << "center_y" << bin.center_residual.y;
    for (int corner = 0; corner < 4; ++corner) {
      storage << ("corner" + std::to_string(corner) + "_x")
              << bin.corner_offsets[corner].x;
      storage << ("corner" + std::to_string(corner) + "_y")
              << bin.corner_offsets[corner].y;
    }
    storage << "}";
  }
  storage << "]";
}

PhaseQuadModel PhaseQuadModel::read(const cv::FileNode& node) {
  if (node.empty() || !node.isMap())
    throw std::runtime_error("calibration profile has no phase quadrilateral model");
  PhaseQuadModel model;
  const int valid = required_int(node, "valid");
  model.covered_bins_ = required_int(node, "covered_bins");
  const cv::FileNode bins = node["bins"];
  if (!bins.isSeq())
    throw std::runtime_error("calibration profile phase bins are invalid");
  for (const auto& stored : bins) {
    if (!stored.isMap())
      throw std::runtime_error("calibration profile has a malformed phase bin");
    Bin bin;
    bin.center_residual.x = required_finite_double(stored, "center_x");
    bin.center_residual.y = required_finite_double(stored, "center_y");
    std::array<cv::Point2f, 4> shape{};
    for (int corner = 0; corner < 4; ++corner) {
      const std::string prefix = "corner" + std::to_string(corner);
      bin.corner_offsets[corner].x =
          required_finite_double(stored, prefix + "_x");
      bin.corner_offsets[corner].y =
          required_finite_double(stored, prefix + "_y");
      shape[corner] = cv::Point2f(bin.corner_offsets[corner]);
    }
    if (!valid_projected_quad(shape))
      throw std::runtime_error("calibration profile has a degenerate phase bin");
    model.bins_.push_back(bin);
  }
  model.valid_ = valid == 1 && !model.bins_.empty() &&
                 model.covered_bins_ > 0 &&
                 model.covered_bins_ <= static_cast<int>(model.bins_.size());
  if (!model.valid_ || valid != 1)
    throw std::runtime_error("calibration profile phase model is not valid");
  return model;
}

void save_calibration_profile(const std::string& path,
                              const CalibrationProfile& profile) {
  if (profile.image_size.width <= 0 || profile.image_size.height <= 0 ||
      profile.sample_count < 5 || !valid_geometry(profile.geometry) ||
      !profile.phase_quad_model.valid())
    throw std::runtime_error("cannot save an invalid calibration profile");
  if (profile.undistorted &&
      (profile.camera_reference_size.width <= 0 ||
       profile.camera_reference_size.height <= 0 ||
       !std::isfinite(profile.camera_matrix(0, 0)) ||
       !std::isfinite(profile.camera_matrix(1, 1)) ||
       profile.camera_matrix(0, 0) <= 0.0 ||
       profile.camera_matrix(1, 1) <= 0.0))
    throw std::runtime_error("cannot save an invalid camera calibration");
  cv::FileStorage storage(path, cv::FileStorage::WRITE);
  if (!storage.isOpened())
    throw std::runtime_error("cannot create calibration profile: " + path);
  storage << "version" << profile.version;
  storage << "image_width" << profile.image_size.width;
  storage << "image_height" << profile.image_size.height;
  storage << "undistorted" << static_cast<int>(profile.undistorted);
  storage << "sample_count" << profile.sample_count;
  storage << "camera" << "{";
  storage << "reference_width" << profile.camera_reference_size.width;
  storage << "reference_height" << profile.camera_reference_size.height;
  storage << "fx" << profile.camera_matrix(0, 0);
  storage << "fy" << profile.camera_matrix(1, 1);
  storage << "cx" << profile.camera_matrix(0, 2);
  storage << "cy" << profile.camera_matrix(1, 2);
  storage << "k1" << profile.distortion[0];
  storage << "k2" << profile.distortion[1];
  storage << "p1" << profile.distortion[2];
  storage << "p2" << profile.distortion[3];
  storage << "k3" << profile.distortion[4];
  storage << "}";
  storage << "geometry" << "{";
  storage << "center_x" << profile.geometry.center.x;
  storage << "center_y" << profile.geometry.center.y;
  storage << "axis_cos_x" << profile.geometry.axis_cos.x;
  storage << "axis_cos_y" << profile.geometry.axis_cos.y;
  storage << "axis_sin_x" << profile.geometry.axis_sin.x;
  storage << "axis_sin_y" << profile.geometry.axis_sin.y;
  storage << "armor_width_scale" << profile.geometry.armor_width_scale;
  storage << "armor_height_px" << profile.geometry.armor_height_px;
  storage << "calibration_samples" << profile.geometry.calibration_samples;
  storage << "calibration_rms_px" << profile.geometry.calibration_rms_px;
  storage << "}";
  storage << "phase_quad_model" << "{";
  profile.phase_quad_model.write(storage);
  storage << "}";
}

CalibrationProfile load_calibration_profile(const std::string& path) {
  cv::FileStorage storage(path, cv::FileStorage::READ);
  if (!storage.isOpened())
    throw std::runtime_error("cannot open calibration profile: " + path);
  CalibrationProfile profile;
  profile.version = required_int(storage.root(), "version");
  profile.image_size.width = required_int(storage.root(), "image_width");
  profile.image_size.height = required_int(storage.root(), "image_height");
  const int undistorted = required_int(storage.root(), "undistorted");
  profile.sample_count = required_int(storage.root(), "sample_count");
  if (undistorted != 0 && undistorted != 1)
    throw std::runtime_error("calibration profile has invalid undistorted flag");
  profile.undistorted = undistorted != 0;
  if (profile.version != CalibrationProfile::kCurrentVersion ||
      profile.image_size.width <= 0 || profile.image_size.height <= 0 ||
      profile.sample_count < 5)
    throw std::runtime_error("unsupported or incomplete calibration profile");
  const cv::FileNode camera = storage["camera"];
  if (camera.empty() || !camera.isMap())
    throw std::runtime_error("calibration camera metadata is missing");
  profile.camera_reference_size.width = required_int(camera, "reference_width");
  profile.camera_reference_size.height = required_int(camera, "reference_height");
  profile.camera_matrix = {
      required_finite_double(camera, "fx"), 0.0,
      required_finite_double(camera, "cx"),
      0.0, required_finite_double(camera, "fy"),
      required_finite_double(camera, "cy"),
      0.0, 0.0, 1.0};
  profile.distortion = {
      required_finite_double(camera, "k1"),
      required_finite_double(camera, "k2"),
      required_finite_double(camera, "p1"),
      required_finite_double(camera, "p2"),
      required_finite_double(camera, "k3")};
  if (profile.undistorted &&
      (profile.camera_reference_size.width <= 0 ||
       profile.camera_reference_size.height <= 0 ||
       profile.camera_matrix(0, 0) <= 0.0 ||
       profile.camera_matrix(1, 1) <= 0.0))
    throw std::runtime_error("calibration camera metadata is invalid");
  const cv::FileNode geometry = storage["geometry"];
  if (geometry.empty() || !geometry.isMap())
    throw std::runtime_error("calibration geometry is missing");
  profile.geometry.center.x = required_finite_double(geometry, "center_x");
  profile.geometry.center.y = required_finite_double(geometry, "center_y");
  profile.geometry.axis_cos.x = required_finite_double(geometry, "axis_cos_x");
  profile.geometry.axis_cos.y = required_finite_double(geometry, "axis_cos_y");
  profile.geometry.axis_sin.x = required_finite_double(geometry, "axis_sin_x");
  profile.geometry.axis_sin.y = required_finite_double(geometry, "axis_sin_y");
  profile.geometry.armor_width_scale =
      required_finite_double(geometry, "armor_width_scale");
  profile.geometry.armor_height_px =
      required_finite_double(geometry, "armor_height_px");
  profile.geometry.calibration_samples = required_int(geometry, "calibration_samples");
  profile.geometry.calibration_rms_px =
      required_finite_double(geometry, "calibration_rms_px");
  if (!valid_geometry(profile.geometry))
    throw std::runtime_error("calibration geometry is degenerate");
  profile.phase_quad_model = PhaseQuadModel::read(storage["phase_quad_model"]);
  return profile;
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
  if (!std::isfinite(config_.prediction_lead_s) ||
      config_.prediction_lead_s <= 0.0)
    throw std::runtime_error("prediction horizon must be resolved before solver construction");
  if (motion_prior_.handover.valid) {
    const cv::Point2d pixel_direction =
        motion_prior_.handover.forward_direction_image;
    const double length = cv::norm(pixel_direction);
    if (length > 1e-6) {
      forward_handover_direction_ = pixel_direction * (1.0 / length);
      handover_direction_valid_ = true;
    }
  }
}

std::array<ProjectedArmor, 3> RigidArmorSolver::project_all(double base_phase) const {
  std::array<ProjectedArmor, 3> projected;
  for (int i = 0; i < 3; ++i) {
    const double phase = base_phase + i * kTwoPi / 3.0;
    projected[i] = phase_quad_model_.project(i + 1, phase, geometry_, model_offset_);
  }
  return projected;
}

void RigidArmorSolver::reset_tracking_state() {
  track_state_ = TrackState::kLost;
  filter_valid_ = false;
  track_confirm_hits_ = 0;
  committed_slot_ = -1;
  last_timestamp_s_ = 0.0;
  last_measurement_time_s_ = 0.0;
  phase_filter_state_ = {0.0, 0.0};
  phase_filter_covariance_ = cv::Matx22d::eye();
  phase_ = 0.0;
  speed_ = 0.0;
  previous_speed_ = 0.0;
  acceleration_ = 0.0;
  missed_frames_ = 0;
  detection_history_valid_ = false;
  previous_detection_time_s_ = 0.0;
  detection_velocity_px_s_ = {};
  detection_velocity_valid_ = false;
  detection_history_.clear();
  display_candidate_slot_ = -1;
  frames_since_handover_ = 0;
  handover_events_.clear();
  physical_speed_samples_.clear();
  physical_speed_valid_ = false;
  physical_speed_ = 0.0;
  physical_acceleration_ = 0.0;
  rotation_direction_votes_.clear();
  rotation_direction_valid_ = false;
  rotation_direction_sign_ = 0;
  committed_mode_ = MotionMode::kInitializing;
  pending_mode_ = MotionMode::kInitializing;
  pending_mode_frames_ = 0;
}

void RigidArmorSolver::initialize_phase_filter(double phase, double timestamp_s) {
  const double phase_std = config_.phase_kf_initial_phase_std_deg * CV_PI / 180.0;
  const double speed_std = config_.phase_kf_initial_speed_std_deg_s * CV_PI / 180.0;
  phase_filter_state_ = {phase, 0.0};
  phase_filter_covariance_ = cv::Matx22d(
      phase_std * phase_std, 0.0, 0.0, speed_std * speed_std);
  filter_valid_ = true;
  phase_ = phase;
  speed_ = 0.0;
  previous_speed_ = 0.0;
  acceleration_ = 0.0;
  last_timestamp_s_ = timestamp_s;
  last_measurement_time_s_ = timestamp_s;
  missed_frames_ = 0;
}

void RigidArmorSolver::predict_phase_filter(double dt) {
  const cv::Matx22d transition(1.0, dt, 0.0, 1.0);
  const double acceleration_std =
      config_.phase_kf_accel_noise_std_deg_s2 * CV_PI / 180.0;
  const double q = acceleration_std * acceleration_std;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;
  const cv::Matx22d process_noise(
      0.25 * dt4 * q, 0.5 * dt3 * q,
      0.5 * dt3 * q, dt2 * q);
  phase_filter_state_ = transition * phase_filter_state_;
  phase_filter_covariance_ = transition * phase_filter_covariance_ *
                             transition.t() + process_noise;
  const double symmetric = 0.5 * (phase_filter_covariance_(0, 1) +
                                  phase_filter_covariance_(1, 0));
  phase_filter_covariance_(0, 0) = std::max(
      config_.covariance_floor, phase_filter_covariance_(0, 0));
  phase_filter_covariance_(1, 1) = std::max(
      config_.covariance_floor, phase_filter_covariance_(1, 1));
  phase_filter_covariance_(0, 1) = symmetric;
  phase_filter_covariance_(1, 0) = symmetric;
  phase_ = phase_filter_state_[0];
  speed_ = phase_filter_state_[1];
}

void RigidArmorSolver::update_phase_filter(double measurement,
                                           double measurement_variance_value,
                                           double innovation) {
  (void)measurement;
  const double innovation_variance = phase_filter_covariance_(0, 0) +
                                     measurement_variance_value;
  const cv::Vec2d gain(phase_filter_covariance_(0, 0) / innovation_variance,
                       phase_filter_covariance_(1, 0) / innovation_variance);
  phase_filter_state_ += gain * innovation;
  const cv::Matx22d joseph_left(1.0 - gain[0], 0.0, -gain[1], 1.0);
  const cv::Matx22d gain_noise(
      gain[0] * gain[0], gain[0] * gain[1],
      gain[1] * gain[0], gain[1] * gain[1]);
  phase_filter_covariance_ =
      joseph_left * phase_filter_covariance_ * joseph_left.t() +
      gain_noise * measurement_variance_value;
  const double symmetric = 0.5 * (phase_filter_covariance_(0, 1) +
                                  phase_filter_covariance_(1, 0));
  phase_filter_covariance_(0, 0) = std::max(
      config_.covariance_floor, phase_filter_covariance_(0, 0));
  phase_filter_covariance_(1, 1) = std::max(
      config_.covariance_floor, phase_filter_covariance_(1, 1));
  phase_filter_covariance_(0, 1) = symmetric;
  phase_filter_covariance_(1, 0) = symmetric;
  phase_ = phase_filter_state_[0];
  speed_ = phase_filter_state_[1];
}

double RigidArmorSolver::measurement_variance(
    const ArmorObservation& observation) const {
  const double quality = std::clamp(static_cast<double>(observation.score), 0.25, 1.0);
  const double sigma = config_.phase_kf_measurement_std_deg * CV_PI / 180.0 / quality;
  return sigma * sigma;
}

MotionMode RigidArmorSolver::classify_mode(double dt) {
  MotionMode candidate;
  const double motion_speed = scene_moving_ ? speed_ : 0.0;
  const double motion_acceleration = scene_moving_ ? acceleration_ : 0.0;
  const double speed_deg = std::abs(motion_speed) * 180.0 / CV_PI;
  const double acceleration_deg = std::abs(motion_acceleration) * 180.0 / CV_PI;
  if (track_state_ == TrackState::kLost) candidate = MotionMode::kLost;
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

SolverOutput RigidArmorSolver::update(const std::vector<ArmorObservation>& observations,
                                      double timestamp_s, bool scene_moving) {
  SolverOutput output;
  scene_moving_ = scene_moving;
  if (detection_history_valid_ &&
      frames_since_handover_ < std::numeric_limits<int>::max())
    ++frames_since_handover_;

  bool timestamp_gap_reset = false;
  double dt = 1.0 / 30.0;
  if (filter_valid_) {
    const double raw_dt = timestamp_s - last_timestamp_s_;
    if (!std::isfinite(timestamp_s) || !std::isfinite(raw_dt) ||
        raw_dt <= 0.0 || raw_dt > config_.tracker_max_dt_s) {
      reset_tracking_state();
      timestamp_gap_reset = true;
    } else {
      dt = raw_dt;
    }
  }
  if (filter_valid_ && track_state_ == TrackState::kDetecting &&
      timestamp_s - last_measurement_time_s_ > config_.track_confirm_max_gap_s) {
    reset_tracking_state();
  }
  if (filter_valid_) predict_phase_filter(dt);
  const double prediction_lead_s = config_.prediction_lead_s;
  const double predicted_phase = phase_;


  int best_observation = -1, best_slot = -1;
  int image_candidate_association_slot = -1;
  double image_candidate_association_cost = std::numeric_limits<double>::infinity();
  double best_candidate_phase = predicted_phase;
  double best_cost = std::numeric_limits<double>::infinity();
  double best_innovation = 0.0;
  double best_innovation_variance = 0.0;
  double best_measurement_variance = 0.0;
  double best_nis = std::numeric_limits<double>::infinity();
  double diagnostic_cost = std::numeric_limits<double>::infinity();
  double diagnostic_innovation = 0.0;
  double diagnostic_innovation_variance = 0.0;
  double diagnostic_nis = std::numeric_limits<double>::infinity();
  for (std::size_t observation_index = 0;
       filter_valid_ && observation_index < observations.size();
       ++observation_index) {
    const auto& observation = observations[observation_index];
    const double measured_ellipse_phase = geometry_.phase_of(observation.center);
    const double observation_variance = measurement_variance(observation);
    const bool reacquiring = track_state_ == TrackState::kTempLost &&
        timestamp_s - last_measurement_time_s_ >
            config_.image_prediction_history_timeout_s;
    int proposed_switch_slot = -1;
    if (!reacquiring && committed_slot_ >= 0 &&
        detection_history_valid_) {
      const double elapsed = timestamp_s - previous_detection_time_s_;
      if (elapsed > 1e-4 &&
          elapsed <= config_.image_prediction_history_timeout_s) {
        const cv::Point2d displacement =
            cv::Point2d(observation.center) -
            cv::Point2d(previous_detection_.center);
        const double nominal_frame_s = prediction_lead_s /
            std::max(1, config_.prediction_lead_frames);
        const double normalized_step =
            cv::norm(displacement) * nominal_frame_s / elapsed;
        if (normalized_step >= config_.image_prediction_jump_min_px &&
            normalized_step <= config_.image_prediction_jump_max_px) {
          bool direction_consistent = true;
          const double velocity_length = cv::norm(detection_velocity_px_s_);
          if (detection_velocity_valid_ && velocity_length > 1.0) {
            const cv::Point2d motion_direction =
                detection_velocity_px_s_ * (1.0 / velocity_length);
            const double displacement_length =
                std::max(1e-9, cv::norm(displacement));
            const double direction_cosine =
                displacement.dot(detection_velocity_px_s_) /
                (displacement_length * velocity_length);
            const double exit_progress =
                (cv::Point2d(previous_detection_.center) - geometry_.center)
                    .dot(motion_direction);
            const double entry_progress =
                (cv::Point2d(observation.center) - geometry_.center)
                    .dot(motion_direction);
            direction_consistent =
                direction_cosine <=
                    config_.image_prediction_jump_direction_cos_max &&
                exit_progress > 0.0 && entry_progress < 0.0;
          }
          if (direction_consistent) {
            int slot_step = 1;
            if (handover_direction_valid_) {
              slot_step = displacement.dot(forward_handover_direction_) >= 0.0
                  ? 1 : -1;
            }
            proposed_switch_slot = (committed_slot_ + slot_step + 3) % 3;
          }
        }
      }
    }
    for (int slot = 0; slot < 3; ++slot) {
      if (committed_slot_ >= 0 && slot != committed_slot_ &&
          !reacquiring && slot != proposed_switch_slot) continue;
      const double raw_base = measured_ellipse_phase - slot * kTwoPi / 3.0;
      const double candidate_phase = predicted_phase +
          std::remainder(raw_base - predicted_phase, kTwoPi);
      const double projected_phase = predicted_phase + slot * kTwoPi / 3.0;
      const auto projected_candidate = phase_quad_model_.project(
          slot + 1, projected_phase,
          geometry_, model_offset_);
      const cv::Point2d association_center = projected_candidate.valid
          ? cv::Point2d(projected_candidate.center)
          : geometry_.point(projected_phase) + model_offset_;
      const double pixel_error = cv::norm(
          association_center - cv::Point2d(observation.center));
      const double innovation = candidate_phase - predicted_phase;
      const double innovation_variance =
          phase_filter_covariance_(0, 0) + observation_variance;
      const double nis = innovation * innovation / innovation_variance;
      const double normalized_pixel =
          pixel_error / config_.association_pixel_sigma_px;
      const double score_penalty = config_.association_score_weight *
          (1.0 - std::clamp(static_cast<double>(observation.score), 0.0, 1.0));
      const double switch_penalty =
          committed_slot_ >= 0 && slot != committed_slot_
              ? config_.association_slot_switch_penalty
              : 0.0;
      const double cost = nis + normalized_pixel * normalized_pixel +
                          score_penalty + switch_penalty;
      if (observation_index == 0 &&
          pixel_error <= config_.max_observation_distance_px &&
          nis <= config_.phase_nis_gate &&
          cost < image_candidate_association_cost) {
        image_candidate_association_cost = cost;
        image_candidate_association_slot = slot;
      }
      if (cost < diagnostic_cost) {
        diagnostic_cost = cost;
        diagnostic_innovation = innovation;
        diagnostic_innovation_variance = innovation_variance;
        diagnostic_nis = nis;
      }
      if (pixel_error > config_.max_observation_distance_px ||
          nis > config_.phase_nis_gate) {
        continue;
      }
      if (cost < best_cost) {
        best_cost = cost; best_observation = static_cast<int>(observation_index);
        best_slot = slot; best_candidate_phase = candidate_phase;
        best_innovation = innovation;
        best_innovation_variance = innovation_variance;
        best_measurement_variance = observation_variance;
        best_nis = nis;
      }
    }
  }
  if (best_observation < 0 && std::isfinite(diagnostic_cost)) {
    best_cost = diagnostic_cost;
    best_innovation = diagnostic_innovation;
    best_innovation_variance = diagnostic_innovation_variance;
    best_nis = diagnostic_nis;
  }
  int candidate_observation = best_observation;
  int accepted_slot = -1;
  bool accepted_measurement = false;
  const bool regular_match = filter_valid_ && best_observation >= 0;

  if (!filter_valid_ && !observations.empty() && std::isfinite(timestamp_s)) {
    const auto best = std::max_element(observations.begin(), observations.end(),
                                      [](const auto& a, const auto& b) { return a.score < b.score; });
    initialize_phase_filter(geometry_.phase_of(best->center), timestamp_s);
    track_state_ = TrackState::kDetecting;
    track_confirm_hits_ = 1;
    best_observation = static_cast<int>(best - observations.begin());
    best_slot = 0;
    best_candidate_phase = phase_;
    candidate_observation = best_observation;
    accepted_slot = best_slot;
    committed_slot_ = accepted_slot;
    accepted_measurement = true;
    best_cost = 0.0;
    best_innovation = 0.0;
    best_innovation_variance = phase_filter_covariance_(0, 0) +
                               measurement_variance(*best);
    best_nis = 0.0;
    if (track_confirm_hits_ >= config_.track_confirm_hits)
      track_state_ = TrackState::kTracking;
  } else if (regular_match) {
    const double measurement_gap_s = timestamp_s - last_measurement_time_s_;
    if (best_slot != committed_slot_ && !handover_direction_valid_ &&
        detection_history_valid_) {
      const cv::Point2d displacement =
          cv::Point2d(observations[static_cast<std::size_t>(best_observation)].center) -
          cv::Point2d(previous_detection_.center);
      const double length = cv::norm(displacement);
      if (length > 1.0) {
        forward_handover_direction_ = displacement * (1.0 / length);
        handover_direction_valid_ = true;
      }
    }
    previous_speed_ = speed_;
    update_phase_filter(best_candidate_phase, best_measurement_variance,
                        best_innovation);
    const double max_speed = config_.max_abs_speed_deg_s * CV_PI / 180.0;
    speed_ = std::clamp(speed_, -max_speed, max_speed);
    if (config_.angular_speed_snap_enabled)
      speed_ = snap_angular_speed(speed_, config_);
    phase_filter_state_[1] = speed_;
    const double instantaneous_acceleration = (speed_ - previous_speed_) / dt;
    acceleration_ = (1.0 - config_.acceleration_gain) * acceleration_ +
                    config_.acceleration_gain * instantaneous_acceleration;
    missed_frames_ = 0;
    accepted_slot = best_slot;
    committed_slot_ = accepted_slot;
    accepted_measurement = true;
    last_measurement_time_s_ = timestamp_s;
    if (track_state_ == TrackState::kDetecting) {
      track_confirm_hits_ = measurement_gap_s <= config_.track_confirm_max_gap_s
          ? track_confirm_hits_ + 1
          : 1;
      if (track_confirm_hits_ >= config_.track_confirm_hits)
        track_state_ = TrackState::kTracking;
    } else {
      track_state_ = TrackState::kTracking;
      track_confirm_hits_ = std::max(track_confirm_hits_, config_.track_confirm_hits);
    }
  } else if (filter_valid_) {
    ++missed_frames_;
    const double miss_age_s = timestamp_s - last_measurement_time_s_;
    if (track_state_ == TrackState::kDetecting &&
        miss_age_s > config_.track_confirm_max_gap_s) {
      reset_tracking_state();
    } else {
      if (track_state_ == TrackState::kTracking)
        track_state_ = TrackState::kTempLost;
      if (track_state_ == TrackState::kTempLost &&
          miss_age_s > config_.track_lost_timeout_s)
        reset_tracking_state();
    }
  }
  if (filter_valid_) last_timestamp_s_ = timestamp_s;

  const int previous_display_candidate_slot = display_candidate_slot_;
  if (accepted_measurement && accepted_slot >= 0) {
    if (display_candidate_slot_ >= 0 &&
        display_candidate_slot_ != accepted_slot)
      frames_since_handover_ = 0;
    display_candidate_slot_ = accepted_slot;
  }

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

  // Keep the raw detector candidate available for diagnostics, but only an
  // observation that passed association and NIS gates may drive display IDs,
  // image history, or the observed green slot.
  const int display_observation = observations.empty() ? -1 : 0;
  const int image_candidate_observation = accepted_measurement
      ? candidate_observation : -1;
  output.candidate_available = display_observation >= 0;
  output.measurement_used = accepted_measurement;
  output.candidate_measurement_used = accepted_measurement &&
                                      candidate_observation == display_observation;
  output.measurement_slot_index = accepted_measurement ? accepted_slot : -1;
  output.candidate_association_slot_index = image_candidate_association_slot;
  output.missed_frames = missed_frames_;
  output.phase_rad = phase_;
  if (physical_speed_valid_ && rotation_direction_valid_)
    physical_speed_ = std::copysign(std::abs(physical_speed_),
                                    static_cast<double>(rotation_direction_sign_));
  output.physical_speed_valid = filter_valid_ && physical_speed_valid_ &&
      rotation_direction_valid_ &&
      (track_state_ == TrackState::kTracking ||
       track_state_ == TrackState::kTempLost);
  output.angular_speed_rad_s = scene_moving_ && output.physical_speed_valid
      ? physical_speed_
      : 0.0;
  output.angular_acceleration_rad_s2 = scene_moving_ && output.physical_speed_valid
      ? physical_acceleration_
      : 0.0;
  output.track_state = track_state_;
  output.track_confirmed = track_state_ == TrackState::kTracking ||
                           track_state_ == TrackState::kTempLost;
  output.phase_gate_passed = accepted_measurement;
  output.timestamp_gap_reset = timestamp_gap_reset;
  output.track_confirm_hits = track_confirm_hits_;
  output.time_since_measurement_s = filter_valid_
      ? std::max(0.0, timestamp_s - last_measurement_time_s_)
      : 0.0;
  output.phase_variance_rad2 = filter_valid_ ? phase_filter_covariance_(0, 0) : 0.0;
  output.speed_variance_rad2_s2 = filter_valid_ ? phase_filter_covariance_(1, 1) : 0.0;
  output.phase_innovation_rad = best_innovation;
  output.phase_innovation_variance = best_innovation_variance;
  output.phase_nis = std::isfinite(best_nis) ? best_nis : 0.0;
  output.association_cost = std::isfinite(best_cost) ? best_cost : 0.0;
  const double phase_sigma = std::sqrt(std::max(0.0, output.phase_variance_rad2));
  const double uncertainty_confidence = std::exp(
      -phase_sigma / std::max(1e-6, 15.0 * CV_PI / 180.0));
  const double state_confidence = output.track_confirmed ? 1.0 :
      (track_state_ == TrackState::kDetecting
           ? static_cast<double>(track_confirm_hits_) /
                 std::max(1, config_.track_confirm_hits)
           : 0.0);
  output.track_confidence = std::clamp(
      uncertainty_confidence * state_confidence, 0.0, 1.0);
  if (display_observation >= 0) {
    output.candidate = observations[static_cast<std::size_t>(display_observation)];
  }

  const bool accepted_display_switch = accepted_measurement &&
      previous_display_candidate_slot >= 0 && accepted_slot >= 0 &&
      previous_display_candidate_slot != accepted_slot;
  // Image-space history is retained only as an association aid for deciding
  // whether a large detector jump is a plausible armor handover.  It no
  // longer owns a second future-position predictor; all displayed future
  // geometry comes from the same filtered phase state used for association.
  if (image_candidate_observation >= 0) {
    const ArmorObservation& current =
        observations[static_cast<std::size_t>(image_candidate_observation)];
    // A display-ID switch is a handover only when the same displayed candidate
    // also passed the Kalman pixel/NIS gates in this frame.
    bool periodic_jump = accepted_measurement &&
        previous_display_candidate_slot >= 0 && accepted_slot >= 0 &&
        previous_display_candidate_slot != accepted_slot;
    bool image_gate_passed = true;
    if (periodic_jump) {
      frames_since_handover_ = 0;
      detection_history_.clear();
    } else if (!detection_history_valid_ ||
        timestamp_s - previous_detection_time_s_ >
            config_.image_prediction_history_timeout_s) {
      detection_velocity_px_s_ = {};
      detection_velocity_valid_ = false;
      detection_history_.clear();
      if (display_candidate_slot_ < 0)
        display_candidate_slot_ = accepted_slot >= 0 ? accepted_slot : std::max(0, best_slot);
    } else {
      const double elapsed = std::max(1e-3, timestamp_s - previous_detection_time_s_);
      const cv::Point2d displacement =
          cv::Point2d(current.center) - cv::Point2d(previous_detection_.center);
      const double nominal_frame_s = prediction_lead_s /
          std::max(1, config_.prediction_lead_frames);
      const double normalized_step = cv::norm(displacement) * nominal_frame_s / elapsed;
      output.image_normalized_step_px = normalized_step;
      if (normalized_step > config_.image_prediction_max_step_px &&
          normalized_step < config_.image_prediction_jump_min_px) {
        image_gate_passed = false;
      } else if (normalized_step >= config_.image_prediction_jump_min_px &&
                 normalized_step <= config_.image_prediction_jump_max_px) {
        bool plausible_handover = true;
        const double velocity_length = cv::norm(detection_velocity_px_s_);
        if (detection_velocity_valid_ && velocity_length > 1.0) {
          const cv::Point2d motion_direction =
              detection_velocity_px_s_ * (1.0 / velocity_length);
          const double displacement_length = std::max(1e-9, cv::norm(displacement));
          const double direction_cosine =
              displacement.dot(detection_velocity_px_s_) /
              (displacement_length * velocity_length);
          const double exit_progress =
              (cv::Point2d(previous_detection_.center) - geometry_.center)
                  .dot(motion_direction);
          const double entry_progress =
              (cv::Point2d(current.center) - geometry_.center).dot(motion_direction);
          plausible_handover =
              direction_cosine <= config_.image_prediction_jump_direction_cos_max &&
              exit_progress > 0.0 && entry_progress < 0.0;
        }
        if (plausible_handover && accepted_measurement) {
          periodic_jump = true;
          const double displacement_length = cv::norm(displacement);
          if (!handover_direction_valid_ && displacement_length > 1.0) {
            forward_handover_direction_ = displacement * (1.0 / displacement_length);
            handover_direction_valid_ = true;
          }
          if (display_candidate_slot_ < 0)
            display_candidate_slot_ = accepted_slot >= 0 ? accepted_slot : std::max(0, best_slot);
          if (accepted_slot >= 0)
            display_candidate_slot_ = accepted_slot;
          frames_since_handover_ = 0;
          detection_history_.clear();
        } else {
          image_gate_passed = false;
        }
      } else if (normalized_step > config_.image_prediction_max_step_px) {
        image_gate_passed = false;
      }
    }

    const bool accepted_periodic_event = accepted_display_switch;
    if (accepted_display_switch) {
      // A slot handover only says that 120 degrees have elapsed.  Its slot
      // index sign is not the sign of the continuous phase state: changing the
      // representative armor adds/subtracts 120 degrees to keep phase
      // continuous.  Vote with the already unwrapped phase-filter speed so the
      // handover-derived magnitude cannot reverse the phase predictor.
      const double direction_vote_threshold =
          config_.stationary_speed_deg_s * CV_PI / 180.0;
      if (std::abs(speed_) >= direction_vote_threshold) {
        rotation_direction_votes_.push_back(speed_ > 0.0 ? 1 : -1);
        while (rotation_direction_votes_.size() >
               static_cast<std::size_t>(config_.physical_direction_window))
          rotation_direction_votes_.pop_front();
        int vote_sum = 0;
        for (const int vote : rotation_direction_votes_) vote_sum += vote;
        if (rotation_direction_votes_.size() >= 3 && vote_sum != 0) {
          rotation_direction_sign_ = vote_sum > 0 ? 1 : -1;
          rotation_direction_valid_ = true;
        }
      }
    }
    if (accepted_periodic_event) {
      const int slot_step = accepted_display_switch
          ? ((accepted_slot - previous_display_candidate_slot + 3) % 3)
          : 0;
      handover_events_.push_back({timestamp_s, slot_step});
      const std::size_t stride = static_cast<std::size_t>(
          config_.physical_speed_handover_stride);
      if (handover_events_.size() > stride) {
        const auto& previous_cycle =
            handover_events_[handover_events_.size() - 1 - stride];
        const double interval_s = timestamp_s - previous_cycle.time_s;
        if (interval_s >= config_.physical_speed_min_interval_s &&
            interval_s <= config_.physical_speed_max_interval_s) {
          const double measured_magnitude =
              angular_speed_from_handover_interval(stride, interval_s);
          const double maximum_magnitude =
              config_.max_abs_speed_deg_s * CV_PI / 180.0;
          if (measured_magnitude <= maximum_magnitude) {
            physical_speed_samples_.push_back(measured_magnitude);
            while (physical_speed_samples_.size() >
                   static_cast<std::size_t>(config_.physical_speed_window))
              physical_speed_samples_.pop_front();
            const double previous_magnitude = std::abs(physical_speed_);
            const double magnitude = median(std::vector<double>(
                physical_speed_samples_.begin(),
                physical_speed_samples_.end()));
            physical_speed_ = rotation_direction_valid_
                ? std::copysign(magnitude,
                                static_cast<double>(rotation_direction_sign_))
                : magnitude;
            if (physical_speed_valid_) {
              const double measured_acceleration =
                  (magnitude - previous_magnitude) /
                  std::max(1e-3, interval_s / stride);
              physical_acceleration_ =
                  (1.0 - config_.physical_speed_acceleration_gain) *
                      physical_acceleration_ +
                  config_.physical_speed_acceleration_gain *
                      measured_acceleration;
            }
            physical_speed_valid_ = true;
          }
        }
      }
      while (handover_events_.size() > stride +
             static_cast<std::size_t>(config_.physical_speed_window) + 1)
        handover_events_.pop_front();
    }

    if (image_gate_passed) {
      output.image_candidate_used = true;
      output.image_handover_detected = periodic_jump;
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
      previous_detection_ = current;
      previous_detection_time_s_ = timestamp_s;
      detection_history_valid_ = true;
    }
  }
  output.detected_slot_index = accepted_measurement ? accepted_slot : -1;
  output.image_track_age_frames = frames_since_handover_;
  const auto projected = project_all(phase_);
  if (accepted_measurement && accepted_slot >= 0) {
    output.reprojection_error_px = cv::norm(
        cv::Point2d(observations[static_cast<std::size_t>(candidate_observation)].center) -
        cv::Point2d(projected[static_cast<std::size_t>(accepted_slot)].center));
  }
  const double orientation = geometry_.axis_cos.x * geometry_.axis_sin.y -
                             geometry_.axis_cos.y * geometry_.axis_sin.x;
  const double image_clockwise_speed = output.angular_speed_rad_s *
      (orientation >= 0.0 ? 1.0 : -1.0);
  if (!scene_moving_) output.direction = "STOP";
  else if (!output.physical_speed_valid) output.direction = "UNKNOWN";
  else if (std::abs(output.angular_speed_rad_s) * 180.0 / CV_PI <
           config_.stationary_speed_deg_s) output.direction = "STOP";
  else output.direction = image_clockwise_speed > 0.0 ? "CW" : "CCW";
  if (accepted_measurement && committed_mode_ == MotionMode::kLost) {
    committed_mode_ = MotionMode::kInitializing;
    pending_mode_ = MotionMode::kInitializing;
    pending_mode_frames_ = 0;
  }
  output.mode = output.track_confirmed
      ? classify_mode(dt)
      : (track_state_ == TrackState::kLost
             ? MotionMode::kLost
             : MotionMode::kInitializing);
  output.model_valid = output.track_confirmed;
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
  const double authoritative_speed =
      scene_moving_ && output.physical_speed_valid ? output.angular_speed_rad_s
                                                   : (scene_moving_ ? speed_ : 0.0);
  if (output.physical_speed_valid) {
    speed_ = authoritative_speed;
    phase_filter_state_[1] = speed_;
  }
  output.future_phase_rad = phase_ + authoritative_speed * output.prediction_lead_s +
      0.5 * prediction_acceleration * output.prediction_lead_s *
          output.prediction_lead_s;
  const auto future_projected = project_all(output.future_phase_rad);
  for (int slot = 0; slot < 3; ++slot) {
    auto& diagnostic = output.future_model_slots[static_cast<std::size_t>(slot)];
    diagnostic.id = slot + 1;
    diagnostic.valid = future_projected[static_cast<std::size_t>(slot)].valid;
    diagnostic.center = future_projected[static_cast<std::size_t>(slot)].center;
    diagnostic.corners = future_projected[static_cast<std::size_t>(slot)].corners;
  }
  const bool current_model_visible = output.model_valid &&
      output.time_since_measurement_s <= config_.track_prediction_visible_s;
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
  // The observed green box is the exact accepted light-band quadrilateral.
  // Rejected candidates remain available only through output.candidate.
  if (accepted_measurement && candidate_observation >= 0 &&
      output.detected_slot_index >= 0 &&
      output.detected_slot_index < 3) {
    auto& destination = output.slots[static_cast<std::size_t>(output.detected_slot_index)];
    const auto& accepted =
        observations[static_cast<std::size_t>(candidate_observation)];
    destination.valid = true;
    destination.source = BoxSource::kObserved;
    destination.center = accepted.center;
    destination.corners = accepted.corners;
  }

  const int target_slot = committed_slot_;
  output.future_slot_index = target_slot;
  output.raw_future_target.id = target_slot + 1;
  if (current_model_visible && target_slot >= 0 && target_slot < 3) {
    const auto& target = future_projected[static_cast<std::size_t>(target_slot)];
    output.raw_future_target.valid = target.valid;
    output.raw_future_target.source =
        target.valid ? BoxSource::kPredicted : BoxSource::kNone;
    output.raw_future_target.center = target.center;
    output.raw_future_target.corners = target.corners;
  }
  output.future_target = output.raw_future_target;
  output.prediction_valid = output.future_target.valid;
  return output;
}
