#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "armor_detector.hpp"
#include "render_policy.hpp"
#include "rigid_model.hpp"

namespace {
constexpr double kTwoPi = 2.0 * CV_PI;

void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void expect_throws(Function function, const std::string& message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

struct TemporaryConfig {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() / "outpost_motion_config_test.yaml";
  ~TemporaryConfig() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
};

LightBar make_light(cv::Point2f center, float length, float width,
                    float angle_from_vertical_deg) {
  LightBar light;
  light.rect = cv::RotatedRect(center, {width, length}, angle_from_vertical_deg);
  light.length = length;
  light.width = width;
  light.long_axis_angle_deg = 90.0F + angle_from_vertical_deg;
  light.area = length * width;
  return light;
}

std::vector<cv::Point2f> polygon(const std::array<cv::Point2f, 4>& corners) {
  return {corners.begin(), corners.end()};
}

void expect_light_inside(const ArmorObservation& armor, const LightBar& light) {
  cv::Point2f vertices[4];
  light.rect.points(vertices);
  const auto contour = polygon(armor.corners);
  for (const auto& vertex : vertices) {
    expect(cv::pointPolygonTest(contour, vertex, true) >= -1.1,
           "observed quadrilateral does not contain a light band");
  }
}

void test_light_band_quad() {
  Config config;
  const auto rectangle = build_armor_observation(
      make_light({80, 100}, 20, 4, 0), make_light({120, 100}, 20, 4, 0),
      config, {200, 200});
  expect(rectangle.has_value(), "parallel light bands should form an armor quad");
  expect(cv::isContourConvex(polygon(rectangle->corners)),
         "parallel light-band quad must be convex");
  expect(std::abs(rectangle->center.x - 100.0F) < 0.1F &&
             std::abs(rectangle->center.y - 100.0F) < 0.1F,
         "armor center should remain the mean of the two light centers");
  expect_light_inside(*rectangle, make_light({80, 100}, 20, 4, 0));
  expect_light_inside(*rectangle, make_light({120, 100}, 20, 4, 0));

  const LightBar left = make_light({78, 99}, 24, 5, -9);
  const LightBar right = make_light({122, 104}, 18, 3, 11);
  const auto trapezoid = build_armor_observation(left, right, config, {220, 220});
  expect(trapezoid.has_value(), "perspective light bands should form a trapezoid");
  expect(cv::isContourConvex(polygon(trapezoid->corners)),
         "perspective light-band quad must be convex");
  expect_light_inside(*trapezoid, left);
  expect_light_inside(*trapezoid, right);
  cv::Point2f left_side = trapezoid->corners[3] - trapezoid->corners[0];
  cv::Point2f right_side = trapezoid->corners[2] - trapezoid->corners[1];
  left_side *= 1.0F / static_cast<float>(cv::norm(left_side));
  right_side *= 1.0F / static_cast<float>(cv::norm(right_side));
  const float side_cross = std::abs(left_side.x * right_side.y - left_side.y * right_side.x);
  expect(side_cross > 0.10F,
         "different light tilts should not be forced into parallel rectangle sides");
}

void test_single_light_is_not_full_armor() {
  Config config;
  config.roi_x_min = 0; config.roi_y_min = 0;
  config.roi_x_max = 1; config.roi_y_max = 1;
  cv::Mat image(180, 180, CV_8UC3, cv::Scalar::all(0));
  const cv::RotatedRect light({90, 90}, {5, 24}, 8);
  cv::Point2f raw[4]; light.points(raw);
  std::vector<cv::Point> vertices;
  for (const auto& point : raw) vertices.emplace_back(cvRound(point.x), cvRound(point.y));
  cv::fillConvexPoly(image, vertices, cv::Scalar(0, 0, 255));
  const DetectionFrame detected = ArmorDetector(config).detect(image);
  expect(detected.armors.empty(), "a single light band must not become a full red armor box");
}

std::vector<ArmorObservation> synthetic_phase_samples(const AffineGeometry& geometry) {
  std::vector<ArmorObservation> samples;
  for (int i = 0; i < 72; ++i) {
    const double phase = kTwoPi * i / 72.0;
    const cv::Point2d center = geometry.point(phase);
    const float width = static_cast<float>(28.0 + 8.0 * std::cos(phase));
    const float height = static_cast<float>(16.0 + 2.0 * std::sin(phase));
    const float shear = static_cast<float>(3.0 * std::sin(phase));
    ArmorObservation sample;
    sample.center = center;
    sample.corners = {
        sample.center + cv::Point2f(-0.5F * width + shear, -0.5F * height),
        sample.center + cv::Point2f(0.5F * width + shear, -0.5F * height),
        sample.center + cv::Point2f(0.5F * width - shear, 0.5F * height),
        sample.center + cv::Point2f(-0.5F * width - shear, 0.5F * height)};
    sample.size = {width, height};
    sample.score = 1.0F;
    samples.push_back(sample);
  }
  return samples;
}

ArmorObservation translated_observation(const ArmorObservation& source,
                                        cv::Point2f translation) {
  ArmorObservation translated = source;
  translated.center += translation;
  for (auto& corner : translated.corners) corner += translation;
  return translated;
}

void test_config_loading_and_validation() {
  TemporaryConfig temporary;
  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "solver_mode" << "affine";
    fs << "camera_enabled" << 1;
    fs << "camera_reference_width" << 668 << "camera_reference_height" << 688;
    fs << "fx" << 800.0 << "fy" << 810.0 << "cx" << 334.0 << "cy" << 344.0;
    fs << "k1" << -0.1 << "k2" << 0.02 << "p1" << 0.001 << "p2" << -0.002
       << "k3" << 0.003;
    fs << "armor_width_m" << 0.140 << "armor_height_m" << 0.060;
    fs << "rotation_radius_m" << 0.310 << "outpost_pitch_deg" << -12.0;
    fs << "max_abs_speed_deg_s" << 150.0;
    fs << "angular_speed_snap_enabled" << 1;
    fs << "angular_speed_snap_rad_s" << 2.20;
    fs << "angular_speed_snap_tolerance_rad_s" << 0.30;
    fs << "future_target_filter_enabled" << 1;
    fs << "future_target_filter_position_gain" << 0.55;
    fs << "future_target_filter_velocity_gain" << 0.18;
    fs << "future_target_filter_max_speed_px_s" << 900.0;
    fs << "prediction_overlap_enter_px" << 6.0;
    fs << "prediction_overlap_exit_px" << 10.0;
    fs << "prediction_overlap_enter_iou" << 0.55;
    fs << "prediction_overlap_exit_iou" << 0.35;
  }
  const Config loaded = Config::load(temporary.path.string());
  expect(loaded.camera_enabled && loaded.camera_reference_width == 668 &&
             loaded.camera_reference_height == 688,
         "camera enable and reference size must load from YAML");
  expect(std::abs(loaded.fx - 800.0) < 1e-9 && std::abs(loaded.k1 + 0.1) < 1e-9,
         "camera intrinsics and distortion must load from YAML");
  expect(std::abs(loaded.rotation_radius_m - 0.310) < 1e-9 &&
             std::abs(loaded.outpost_pitch_deg + 12.0) < 1e-9,
         "physical radius and pitch must load from YAML");
  expect(loaded.angular_speed_snap_enabled &&
             std::abs(loaded.angular_speed_snap_rad_s - 2.20) < 1e-9 &&
             std::abs(loaded.angular_speed_snap_tolerance_rad_s - 0.30) < 1e-9,
         "angular-speed snap settings must load from YAML");
  expect(loaded.future_target_filter_enabled &&
             std::abs(loaded.future_target_filter_position_gain - 0.55) < 1e-9 &&
             std::abs(loaded.future_target_filter_velocity_gain - 0.18) < 1e-9 &&
             std::abs(loaded.future_target_filter_max_speed_px_s - 900.0) < 1e-9,
         "future-target state-filter gains must load from YAML");
  expect(std::abs(loaded.prediction_overlap_enter_px - 6.0) < 1e-9 &&
             std::abs(loaded.prediction_overlap_exit_px - 10.0) < 1e-9 &&
             std::abs(loaded.prediction_overlap_enter_iou - 0.55) < 1e-9 &&
             std::abs(loaded.prediction_overlap_exit_iou - 0.35) < 1e-9,
         "prediction overlap hysteresis must load from YAML");

  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "camera_enabled" << 2;
  }
  expect_throws([&] { Config::load(temporary.path.string()); },
                "non-boolean camera_enabled must be rejected");

  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "image_prediction_jump_min_px" << 60.0;
    fs << "image_prediction_jump_max_px" << 50.0;
  }
  expect_throws([&] { Config::load(temporary.path.string()); },
                "image handover maximum must exceed its minimum");

  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "image_prediction_max_step_px" <<
        std::numeric_limits<double>::quiet_NaN();
  }
  expect_throws([&] { Config::load(temporary.path.string()); },
                "non-finite image prediction thresholds must be rejected");

  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "future_target_filter_enabled" << 2;
  }
  expect_throws([&] { Config::load(temporary.path.string()); },
                "non-boolean future-target filter enable must be rejected");

  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "future_target_filter_position_gain" << 0.0;
  }
  expect_throws([&] { Config::load(temporary.path.string()); },
                "zero future-target position gain must be rejected");

  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "future_target_filter_max_speed_px_s" << 0.0;
  }
  expect_throws([&] { Config::load(temporary.path.string()); },
                "zero future-target speed limit must be rejected");

  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "prediction_overlap_enter_px" << 12.0;
    fs << "prediction_overlap_exit_px" << 8.0;
  }
  expect_throws([&] { Config::load(temporary.path.string()); },
                "overlap exit distance below enter distance must be rejected");

  {
    cv::FileStorage fs(temporary.path.string(), cv::FileStorage::WRITE);
    fs << "prediction_overlap_enter_iou" << 0.30;
    fs << "prediction_overlap_exit_iou" << 0.40;
  }
  expect_throws([&] { Config::load(temporary.path.string()); },
                "overlap exit IoU above enter IoU must be rejected");

  Config second_horizon;
  second_horizon.prediction_lead_s = 0.20;
  second_horizon.resolve_prediction_horizon(30.0);
  expect(second_horizon.prediction_lead_frames == 6 &&
             std::abs(second_horizon.prediction_lead_s - 0.20) < 1e-9,
         "second-based prediction horizon must resolve to the same frame horizon");
  Config frame_horizon;
  frame_horizon.prediction_lead_frames = 4;
  frame_horizon.prediction_lead_s = 0.0;
  frame_horizon.resolve_prediction_horizon(25.0);
  expect(std::abs(frame_horizon.prediction_lead_s - 0.16) < 1e-9,
         "frame-based prediction horizon must resolve to the same second horizon");

  const Config video_config = Config::load(
      std::string(OUTPOST_SOURCE_DIR) + "/config/video.yaml");
  expect(!video_config.camera_enabled && video_config.camera_reference_width == 668 &&
             video_config.camera_reference_height == 688,
         "repository video config must use this video's dimensions with calibration disabled");
  expect(video_config.rotation_radius_m == 0.0 && video_config.outpost_pitch_deg == 0.0,
         "unknown 3D values must not copy another project's constants");
  expect(!video_config.angular_speed_snap_enabled &&
             video_config.angular_speed_snap_rad_s == 0.0 &&
             video_config.angular_speed_snap_tolerance_rad_s == 0.0,
         "video config must not carry a disabled preset angular speed");
  expect(std::abs(video_config.image_prediction_history_timeout_s - 0.15) < 1e-9 &&
             std::abs(video_config.image_prediction_handover_hold_s - 0.90) < 1e-9 &&
             std::abs(video_config.image_prediction_max_step_px - 35.0) < 1e-9 &&
             std::abs(video_config.image_prediction_jump_max_px - 180.0) < 1e-9 &&
             !video_config.image_prediction_handover_prior_enabled &&
             video_config.image_prediction_handover_vote_threshold == 2,
         "video config must expose the measured image-continuity thresholds");
  expect(video_config.future_target_filter_enabled &&
             std::abs(video_config.future_target_filter_position_gain - 0.55) < 1e-9 &&
             std::abs(video_config.future_target_filter_velocity_gain - 0.08) < 1e-9 &&
             std::abs(video_config.future_target_filter_shape_gain - 0.45) < 1e-9 &&
             std::abs(video_config.future_target_filter_reset_distance_px - 60.0) < 1e-9 &&
             std::abs(video_config.future_target_filter_max_speed_px_s - 1050.0) < 1e-9,
         "video config must expose the measured continuous prediction filter");
  expect(video_config.track_confirm_hits == 3 &&
             std::abs(video_config.track_prediction_visible_s - 0.90) < 1e-9 &&
             std::abs(video_config.track_lost_timeout_s - 1.50) < 1e-9 &&
             std::abs(video_config.tracker_max_dt_s - 0.20) < 1e-9 &&
             std::abs(video_config.phase_kf_measurement_std_deg - 10.0) < 1e-9 &&
             std::abs(video_config.phase_nis_gate - 25.0) < 1e-9,
         "video config must expose lifecycle and covariance-filter parameters");
  expect(video_config.physical_speed_handover_stride == 3 &&
             video_config.physical_speed_window == 5 &&
             video_config.physical_direction_window == 15,
         "video config must expose causal handover-speed estimation parameters");
}

void test_no_speed_prior_injection() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.angular_speed_snap_enabled = true;
  config.angular_speed_snap_rad_s = 1.10;
  config.angular_speed_snap_tolerance_rad_s = 0.05;
  config.resolve_prediction_horizon(30.0);

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape = calibrate_phase_quad_model(samples, geometry, config);
  MotionPrior prior;
  const SolverOutput output =
      RigidArmorSolver(config, geometry, shape, prior).update({samples.front()}, 0.0, true);
  expect(!output.physical_speed_valid &&
             std::abs(output.angular_speed_rad_s) < 1e-9,
         "a configured speed snap must not initialize runtime angular speed");
  expect(std::abs(output.future_phase_rad - output.phase_rad) < 1e-12 &&
             output.direction == "UNKNOWN" &&
             std::abs(output.angular_acceleration_rad_s2) < 1e-12,
         "no causal speed evidence must remain stationary in phase and explicitly unknown");
}

void test_robust_image_velocity_prediction() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.max_observation_distance_px = 55.0;
  config.prediction_lead_frames = 3;
  config.prediction_lead_s = 0.10;
  config.image_prediction_window_frames = 5;
  config.image_prediction_min_samples = 3;
  config.image_prediction_velocity_gain = 1.0;
  config.image_prediction_history_timeout_s = 0.15;
  config.image_prediction_max_step_px = 35.0;
  config.image_prediction_jump_min_px = 60.0;
  config.image_prediction_jump_max_px = 180.0;
  config.image_prediction_jump_direction_cos_max = -0.5;

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  // Keep this test focused on image-velocity history expiry. Timestamp-gap
  // reset behavior is covered independently by the tracker lifecycle tests.
  config.tracker_max_dt_s = 1.0;
  geometry.axis_sin = {0, 24};
  const auto phase_samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape = calibrate_phase_quad_model(phase_samples, geometry, config);
  expect(shape.valid(), "robust image prediction test needs a valid phase model");

  MotionPrior prior;
  RigidArmorSolver solver(config, geometry, shape, prior);

  // Ground truth moves at 60 px/s.  The fourth center has an 8 px outlier;
  // both adjacent steps still fall inside the ordinary-track gate, so a
  // last-two-frames estimator would point sharply backwards on the fifth
  // frame.  Irregular timestamps also verify that the fit uses elapsed time.
  constexpr double velocity_x_px_s = 60.0;
  const std::array<double, 5> times{0.000, 0.020, 0.055, 0.090, 0.125};
  SolverOutput output;
  for (std::size_t i = 0; i < times.size(); ++i) {
    float x = static_cast<float>(velocity_x_px_s * times[i]);
    if (i == 3) x += 8.0F;
    const ArmorObservation observation =
        translated_observation(phase_samples.front(), {x, 0.0F});
    output = solver.update({observation}, times[i], true);
  }

  expect(output.image_motion_prediction_used,
         "five valid centers must enable image-motion prediction");
  expect(output.future_target.valid,
         "robust image-motion prediction must produce a future target");
  const double expected_future_x = output.candidate.center.x +
      velocity_x_px_s * config.prediction_lead_s;
  expect(std::abs(output.future_target.center.x - expected_future_x) < 0.75,
         "Theil-Sen velocity must preserve the constant trend despite one outlier");
  expect(std::abs(output.future_target.center.y - output.candidate.center.y) < 0.25,
         "a horizontal robust track must not invent vertical velocity");
  expect(output.future_target.center.x > output.candidate.center.x + 4.0F,
         "future target must advance with the robust trend, not the last outlier step");

  const ArmorObservation rejected =
      translated_observation(phase_samples.front(), {0.0F, 100.0F});
  const SolverOutput gated = solver.update({rejected}, 0.150, true);
  expect(gated.candidate_available && !gated.measurement_used,
         "a medium vertical outlier must fail the state-assimilation gate");
  expect(!gated.image_candidate_used && !gated.image_handover_detected,
         "a 100 px off-trajectory outlier must not be classified as a handover");
  expect(gated.image_motion_prediction_used && gated.image_prediction_coasting,
         "a gated candidate must coast only on the previously accepted image velocity");
  const double gated_future_x = phase_samples.front().center.x +
      velocity_x_px_s * (0.150 + config.prediction_lead_s);
  expect(std::abs(gated.future_target.center.x - gated_future_x) < 0.75,
         "the rejected candidate must not contaminate trusted short-term extrapolation");

  const ArmorObservation resumed = translated_observation(
      phase_samples.front(), {static_cast<float>(velocity_x_px_s * 0.160), 0.0F});
  const SolverOutput after_gate = solver.update({resumed}, 0.160, true);
  expect(after_gate.measurement_used && after_gate.image_motion_prediction_used,
         "the accepted track must resume image prediction after a gated outlier");
  expect(after_gate.image_candidate_used, "the resumed image track must pass continuity gating");
  expect(!after_gate.image_prediction_coasting,
         "an accepted measurement must leave coasting mode immediately");
  const double resumed_future_x = after_gate.candidate.center.x +
      velocity_x_px_s * config.prediction_lead_s;
  expect(std::abs(after_gate.future_target.center.x - resumed_future_x) < 0.75,
         "a gated outlier must not contaminate the accepted velocity history");

  const int age_before_timeout = after_gate.image_track_age_frames;
  const ArmorObservation after_timeout_observation = translated_observation(
      phase_samples.front(), {static_cast<float>(velocity_x_px_s * 0.400), 0.0F});
  const SolverOutput after_timeout =
      solver.update({after_timeout_observation}, 0.400, true);
  expect(after_timeout.image_track_age_frames > age_before_timeout,
         "velocity-history timeout must not reset age since the last handover");
}

void test_continuous_future_filter_and_render_hysteresis() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.max_observation_distance_px = 1000.0;
  config.prediction_lead_frames = 3;
  config.prediction_lead_s = 0.10;
  config.image_prediction_window_frames = 5;
  config.image_prediction_min_samples = 3;
  config.image_prediction_velocity_gain = 1.0;
  config.future_target_filter_enabled = true;
  config.future_target_filter_position_gain = 0.55;
  config.future_target_filter_velocity_gain = 0.08;
  config.future_target_filter_shape_gain = 0.45;
  config.future_target_filter_reset_distance_px = 60.0;
  config.future_target_filter_max_dt_s = 0.20;

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto phase_samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape =
      calibrate_phase_quad_model(phase_samples, geometry, config);
  MotionPrior prior;
  RigidArmorSolver solver(config, geometry, shape, prior);

  std::vector<double> raw_x;
  std::vector<double> filtered_x;
  for (int frame = 0; frame < 14; ++frame) {
    const float jitter = frame % 2 == 0 ? -3.5F : 3.5F;
    const cv::Point2f translation(
        static_cast<float>(2.0 * frame) + jitter, 0.0F);
    const ArmorObservation observation =
        translated_observation(phase_samples.front(), translation);
    const SolverOutput output =
        solver.update({observation}, frame / 30.0, true);
    expect(output.candidate_available,
           "continuous filter test requires a detector candidate");
    const auto observed = std::find_if(
        output.slots.begin(), output.slots.end(),
        [](const ArmorSlotOutput& slot) {
          return slot.source == BoxSource::kObserved;
        });
    expect(observed != output.slots.end() &&
               cv::norm(observed->center - observation.center) < 0.01,
           "future filtering must never move the green detector box");
    if (frame >= 4 && output.image_motion_prediction_used) {
      expect(output.raw_future_target.valid && output.future_target.valid &&
                 output.future_target_filter_used,
             "enabled continuous filtering must expose raw and filtered targets");
      expect(cv::norm(output.future_target_filter_velocity_px_s) <=
                 config.future_target_filter_max_speed_px_s + 1e-6,
             "future-target filter velocity must stay inside its configured bound");
      raw_x.push_back(output.raw_future_target.center.x);
      filtered_x.push_back(output.future_target.center.x);
    }
  }
  expect(raw_x.size() >= 7,
         "continuous filter test needs enough image-motion predictions");
  double raw_second_difference = 0.0;
  double filtered_second_difference = 0.0;
  for (std::size_t index = 2; index < raw_x.size(); ++index) {
    raw_second_difference += std::abs(
        raw_x[index] - 2.0 * raw_x[index - 1] + raw_x[index - 2]);
    filtered_second_difference += std::abs(
        filtered_x[index] - 2.0 * filtered_x[index - 1] +
        filtered_x[index - 2]);
  }
  expect(filtered_second_difference < raw_second_difference,
         "alpha-beta prediction state must reduce frame-to-frame acceleration jitter");

  SolverOutput render_output;
  render_output.candidate_available = true;
  render_output.candidate = phase_samples.front();
  render_output.future_target.valid = true;
  render_output.future_target.id = 1;
  render_output.future_target.center =
      render_output.candidate.center + cv::Point2f(5.0F, 0.0F);
  for (int corner = 0; corner < 4; ++corner) {
    render_output.future_target.corners[corner] =
        render_output.candidate.corners[corner] + cv::Point2f(5.0F, 0.0F);
  }
  PredictionRenderState render_state;
  auto decision =
      evaluate_prediction_render(render_output, config, &render_state);
  expect(decision.draw && decision.high_overlap,
         "a valid overlapping prediction must stay drawn in dimmed style");

  render_output.future_target.center =
      render_output.candidate.center + cv::Point2f(12.0F, 0.0F);
  for (int corner = 0; corner < 4; ++corner) {
    render_output.future_target.corners[corner] =
        render_output.candidate.corners[corner] + cv::Point2f(12.0F, 0.0F);
  }
  decision = evaluate_prediction_render(render_output, config, &render_state);
  expect(decision.draw && decision.high_overlap,
         "overlap style must remain stable until both exit thresholds are met");

  render_output.candidate_available = false;
  decision = evaluate_prediction_render(render_output, config, &render_state);
  expect(decision.draw && decision.high_overlap,
         "a short detector dropout must preserve overlap style without hiding");
  render_output.candidate_available = true;

  render_output.future_target.center =
      render_output.candidate.center + cv::Point2f(24.0F, 0.0F);
  for (int corner = 0; corner < 4; ++corner) {
    render_output.future_target.corners[corner] =
        render_output.candidate.corners[corner] + cv::Point2f(24.0F, 0.0F);
  }
  decision = evaluate_prediction_render(render_output, config, &render_state);
  expect(decision.draw && !decision.high_overlap,
         "separated prediction must leave dimmed overlap style without hiding");

  render_output.future_target.valid = false;
  decision = evaluate_prediction_render(render_output, config, &render_state);
  expect(!decision.draw && !render_state.high_overlap,
         "invalid prediction must clear overlap state instead of extending lifetime");
}

void test_periodic_handover_identity_and_future_id() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.max_observation_distance_px = 1000.0;
  config.phase_nis_gate = 1.0e6;
  config.prediction_lead_frames = 3;
  config.prediction_lead_s = 0.10;
  config.image_prediction_window_frames = 5;
  config.image_prediction_min_samples = 3;
  config.image_prediction_velocity_gain = 1.0;
  config.image_prediction_history_timeout_s = 0.15;
  config.image_prediction_max_step_px = 35.0;
  config.image_prediction_jump_min_px = 60.0;
  config.image_prediction_jump_max_px = 180.0;
  config.image_prediction_jump_direction_cos_max = -0.5;
  config.image_prediction_transition_y_tolerance_px = 8.0;
  config.image_prediction_wrap_margin_px = 8.0;
  config.future_target_filter_enabled = true;
  config.future_target_filter_position_gain = 0.55;
  config.future_target_filter_velocity_gain = 0.08;

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto phase_samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape = calibrate_phase_quad_model(phase_samples, geometry, config);
  MotionPrior prior;
  RigidArmorSolver solver(config, geometry, shape, prior);
  const auto at_x = [&](float x) {
    return translated_observation(
        phase_samples.front(), {x - phase_samples.front().center.x, 0.0F});
  };

  const std::array<float, 4> approach{120.0F, 100.0F, 80.0F, 60.0F};
  SolverOutput state;
  for (std::size_t i = 0; i < approach.size(); ++i)
    state = solver.update({at_x(approach[i])}, static_cast<double>(i) / 30.0, true);
  expect(state.detected_slot_index == 0, "continuous motion must keep the initial A1 identity");

  state = solver.update({at_x(140.0F)}, 4.0 / 30.0, true);
  expect(state.image_handover_detected && state.image_candidate_used,
         "opposite cross-center jump must be classified as a periodic handover");
  expect(state.detected_slot_index == 1,
         "the first confirmed handover must atomically advance A1 to A2");
  expect(state.image_track_age_frames == 0, "a confirmed handover must reset track age");
  expect(state.future_target_filter_reset &&
             cv::norm(state.future_target.center -
                      state.raw_future_target.center) < 0.01,
         "a physical armor-ID change must atomically reset the prediction state");

  const std::array<float, 4> next_approach{120.0F, 100.0F, 80.0F, 65.0F};
  bool future_handover_reset = false;
  for (std::size_t i = 0; i < next_approach.size(); ++i) {
    state = solver.update({at_x(next_approach[i])}, (5.0 + i) / 30.0, true);
    if (state.image_future_handover && state.future_target_filter_reset) {
      future_handover_reset =
          cv::norm(state.future_target.center -
                   state.raw_future_target.center) < 0.01;
    }
  }
  expect(state.image_future_handover && state.image_motion_prediction_used,
         "learned exit/entry geometry must predict an imminent handover");
  expect(state.detected_slot_index == 1 && state.future_target.id == 3,
         "current A2 must remain stable while the future target advances to A3");
  expect(future_handover_reset,
         "future handover must jump directly to the next real armor, never interpolate");
}

void test_learned_handover_cold_start() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.max_observation_distance_px = 1000.0;
  config.prediction_lead_frames = 3;
  config.prediction_lead_s = 0.10;
  config.image_prediction_handover_prior_enabled = true;
  config.image_prediction_handover_vote_threshold = 2;

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto phase_samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape = calibrate_phase_quad_model(phase_samples, geometry, config);
  const auto at_x = [&](float x) {
    return translated_observation(
        phase_samples.front(), {x - phase_samples.front().center.x, 0.0F});
  };

  MotionPrior prior;
  prior.handover.valid = true;
  prior.handover.forward_direction_image = {1.0, 0.0};
  for (auto& slot : prior.handover.slots) {
    slot.valid = true;
    slot.transition_samples = 3;
    slot.exit_template = at_x(60.0F);
    slot.entry_template = at_x(140.0F);
    slot.progress_age_weight_px = 0.0;
    slot.progress_threshold = 70.0;
    slot.age_threshold_frames = 0;
  }

  const SolverOutput output =
      RigidArmorSolver(config, geometry, shape, prior).update({at_x(65.0F)}, 0.0, true);
  expect(output.image_future_handover && output.image_handover_vote_count == 2,
         "calibrated progress and age votes must support the first handover without velocity");
  expect(output.image_handover_progress_vote && output.image_handover_age_vote &&
             !output.image_handover_spatial_vote,
         "cold-start handover must not invent a spatial vote without image velocity");
  expect(output.future_target.valid && output.future_target.id == 2 &&
             std::abs(output.future_target.center.x - 140.0F) < 0.5F,
         "cold-start handover must use the learned next-armor entry template and ID");
}

void test_candidate_measurement_semantics() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.max_observation_distance_px = 55.0;
  config.resolve_prediction_horizon(30.0);
  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape = calibrate_phase_quad_model(samples, geometry, config);
  MotionPrior prior;
  RigidArmorSolver solver(config, geometry, shape, prior);
  const SolverOutput initial = solver.update({samples.front()}, 0.0, true);
  expect(initial.measurement_used && initial.candidate_measurement_used,
         "an accepted primary candidate must be labeled USED");

  ArmorObservation display_outlier = translated_observation(
      samples.front(), {0.0F, 45.0F});
  display_outlier.score = samples.front().score + 1.0F;
  ArmorObservation phase_match = samples.front();
  phase_match.score = samples.front().score;
  const SolverOutput split_candidate = solver.update(
      {display_outlier, phase_match}, 1.0 / 30.0, true);
  expect(split_candidate.measurement_used &&
             !split_candidate.candidate_measurement_used,
         "phase use of a secondary candidate must not label the displayed candidate USED");
  expect(cv::norm(split_candidate.candidate.center - display_outlier.center) < 0.1,
         "the green candidate must remain the detector's highest-confidence observation");
}

void test_phase_quad_model_and_render_sources() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.prediction_lead_frames = 3;
  config.prediction_lead_s = 3.0 / 30.0;
  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape = calibrate_phase_quad_model(samples, geometry, config);
  expect(shape.valid(), "phase quadrilateral model should calibrate");
  expect(shape.covered_bins() >= 12, "phase quadrilateral model needs sufficient coverage");

  const auto before_wrap = shape.project(1, -1e-4, geometry, {});
  const auto after_wrap = shape.project(1, kTwoPi - 1e-4, geometry, {});
  expect(before_wrap.valid && after_wrap.valid, "phase projections must be valid");
  for (int i = 0; i < 4; ++i)
    expect(cv::norm(before_wrap.corners[i] - after_wrap.corners[i]) < 0.1,
           "phase model must interpolate continuously across 2pi");
  for (int degree = 0; degree < 360; ++degree) {
    const auto projected = shape.project(1, degree * CV_PI / 180.0, geometry, {});
    expect(projected.valid && cv::isContourConvex(polygon(projected.corners)),
           "every learned phase projection must remain a valid convex quadrilateral");
  }

  MotionPrior prior;
  RigidArmorSolver solver(config, geometry, shape, prior);
  const SolverOutput observed = solver.update({samples.front()}, 0.0, true);
  expect(observed.candidate_available && observed.detected_slot_index == 0,
         "current detector candidate should be exposed separately as slot A1");
  expect(observed.measurement_used && observed.candidate_measurement_used &&
             observed.measurement_slot_index == 0,
         "first valid detector candidate should initialize slot A1 state");
  expect(cv::norm(observed.candidate.center - samples.front().center) < 0.1,
         "green detector candidate must preserve the current light-band observation");
  expect(observed.prediction_lead_frames == config.prediction_lead_frames &&
             std::abs(observed.prediction_lead_s - config.prediction_lead_s) < 1e-9,
         "solver output must report the configured future horizon");
  int detected_now = 0, model_now = 0;
  for (const auto& slot : observed.slots) {
    detected_now += slot.valid && slot.source == BoxSource::kObserved;
    model_now += slot.valid && slot.source == BoxSource::kPredicted;
  }
  expect(detected_now == 1 && model_now == 2,
         "current slots must contain one exact detection and two current model boxes");
  expect(observed.future_target.valid &&
             observed.future_target.source == BoxSource::kPredicted,
         "the future target must be separate from the current green slots");
  expect(observed.raw_future_target.valid,
         "solver must expose the unfiltered future target for diagnostics");
  expect(std::abs(observed.future_phase_rad - observed.phase_rad) < 1e-9,
         "the first frame must not invent angular velocity from an offline prior");
  expect(cv::norm(observed.future_target.center - observed.candidate.center) < 0.1,
         "cold-start prediction must remain at the observation until motion is measured");

  const SolverOutput handover = solver.update({samples[24]}, 1.0 / 30.0, true);
  expect(handover.measurement_used && handover.measurement_slot_index == 1,
         "a 120-degree handover must associate the accepted observation with A2");
  expect(handover.detected_slot_index == handover.measurement_slot_index,
         "an accepted observation must use the same fixed ID for filtering and display");
  expect(handover.future_target.id == handover.measurement_slot_index + 1,
         "future prediction must inherit the last accepted fixed ID");

  const SolverOutput gated = solver.update({samples[36]}, 2.0 / 30.0, true);
  expect(gated.candidate_available && !gated.measurement_used,
         "an out-of-gate light-band quad must remain a visible current candidate");
  expect(gated.detected_slot_index == handover.measurement_slot_index,
         "a gated candidate must keep the last accepted fixed ID instead of jittering");
  expect(cv::norm(gated.candidate.center - samples[36].center) < 0.1,
         "gating must not replace the current detector quadrilateral with a model box");
  expect(!gated.image_motion_prediction_used,
         "a gated candidate must not update the trusted image-motion predictor");
  expect(!gated.image_prediction_coasting,
         "coasting requires a previously valid image velocity");
  detected_now = 0;
  for (const auto& slot : gated.slots)
    detected_now += slot.valid && slot.source == BoxSource::kObserved;
  expect(detected_now == 1,
         "a gated current detection must remain the exact visible green recognition box");
  expect(gated.future_target.valid,
         "a gated current detection must retain a separate future target");

  const SolverOutput predicted = solver.update({}, 3.0 / 30.0, false);
  expect(!predicted.candidate_available && predicted.detected_slot_index == -1,
         "a missing detector candidate must remain distinct from model prediction");
  detected_now = 0; model_now = 0;
  for (const auto& slot : predicted.slots) {
    detected_now += slot.valid && slot.source == BoxSource::kObserved;
    model_now += slot.valid && slot.source == BoxSource::kPredicted;
  }
  expect(detected_now == 0 && model_now == 3,
         "missing detector candidate must expose three current model boxes");
  expect(predicted.future_target.valid,
         "short missing intervals should keep a separate future target");

  SolverOutput lost = predicted;
  const int loss_frames =
      static_cast<int>(std::ceil(config.track_lost_timeout_s * 30.0)) + 1;
  for (int i = 0; i < loss_frames; ++i)
    lost = solver.update({}, (i + 4) / 30.0, false);
  expect(!lost.model_valid, "prediction must expire after the elapsed-time loss timeout");
  for (const auto& slot : lost.slots) expect(!slot.valid, "LOST state must not render armor boxes");
  expect(!lost.future_target.valid, "LOST state must not render a future target");
  expect(!lost.raw_future_target.valid && !lost.future_target_filter_used,
         "LOST state must clear both raw and filtered future targets");
}
void test_tracker_lifecycle_uses_elapsed_time() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.track_confirm_hits = 3;
  config.track_confirm_max_gap_s = 0.10;
  config.track_prediction_visible_s = 0.15;
  config.track_lost_timeout_s = 0.30;
  config.tracker_max_dt_s = 0.50;
  config.max_observation_distance_px = 1000.0;
  config.phase_nis_gate = 1e6;
  config.prediction_lead_s = 0.10;
  config.resolve_prediction_horizon(30.0);

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape =
      calibrate_phase_quad_model(samples, geometry, config);
  MotionPrior prior;
  RigidArmorSolver solver(config, geometry, shape, prior);

  const auto first = solver.update({samples[0]}, 0.00, true);
  expect(first.track_state == TrackState::kDetecting &&
             first.track_confirm_hits == 1 &&
             !first.track_confirmed && !first.model_valid,
         "first hit must start an unconfirmed track");
  const auto second = solver.update({samples[0]}, 0.04, true);
  expect(second.track_state == TrackState::kDetecting &&
             second.track_confirm_hits == 2,
         "second timely hit must advance confirmation");
  const auto confirmed = solver.update({samples[0]}, 0.08, true);
  expect(confirmed.track_state == TrackState::kTracking &&
             confirmed.track_confirm_hits == 3 &&
             confirmed.track_confirmed && confirmed.model_valid,
         "third timely hit must confirm the track");

  const auto short_miss = solver.update({}, 0.12, true);
  expect(short_miss.track_state == TrackState::kTempLost &&
             short_miss.track_confirmed && short_miss.model_valid &&
             short_miss.missed_frames == 1 &&
             std::abs(short_miss.time_since_measurement_s - 0.04) < 1e-9 &&
             short_miss.future_target.valid,
         "short dropout must coast as TEMP_LOST");
  const auto hidden = solver.update({}, 0.25, true);
  expect(hidden.track_state == TrackState::kTempLost &&
             hidden.model_valid && !hidden.future_target.valid &&
             std::none_of(hidden.slots.begin(), hidden.slots.end(),
                          [](const auto& slot) { return slot.valid; }),
         "render visibility must expire before the retained track identity");
  const auto lost = solver.update({}, 0.40, true);
  expect(lost.track_state == TrackState::kLost &&
             !lost.track_confirmed && !lost.model_valid &&
             lost.track_confirm_hits == 0 && lost.missed_frames == 0 &&
             lost.time_since_measurement_s == 0.0 &&
             lost.phase_variance_rad2 == 0.0 &&
             lost.speed_variance_rad2_s2 == 0.0 &&
             !lost.timestamp_gap_reset,
         "loss timeout must atomically clear the track without claiming a dt reset");

  RigidArmorSolver confirmation_solver(config, geometry, shape, prior);
  confirmation_solver.update({samples[0]}, 1.00, true);
  const auto late_hit =
      confirmation_solver.update({samples[0]}, 1.11, true);
  expect(late_hit.measurement_used &&
             late_hit.track_state == TrackState::kDetecting &&
             late_hit.track_confirm_hits == 1 &&
             !late_hit.timestamp_gap_reset,
         "a late second hit must restart rather than continue confirmation");
}

void test_timestamp_discontinuity_resets_filter() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.track_confirm_hits = 2;
  config.track_confirm_max_gap_s = 0.10;
  config.tracker_max_dt_s = 0.20;
  config.max_observation_distance_px = 1000.0;
  config.phase_nis_gate = 1e6;
  config.prediction_lead_s = 0.10;
  config.resolve_prediction_horizon(30.0);

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape =
      calibrate_phase_quad_model(samples, geometry, config);
  MotionPrior prior;
  RigidArmorSolver solver(config, geometry, shape, prior);

  solver.update({samples[0]}, 0.00, true);
  const auto before = solver.update({samples[1]}, 0.05, true);
  expect(before.track_confirmed &&
             before.future_phase_rad > before.phase_rad + 1e-4,
         "fixture must learn positive KF velocity before reset");

  const auto gap = solver.update({samples[3]}, 0.30, true);
  expect(gap.timestamp_gap_reset && gap.measurement_used &&
             gap.phase_gate_passed &&
             gap.track_state == TrackState::kDetecting &&
             gap.track_confirm_hits == 1 &&
             !gap.track_confirmed && !gap.model_valid &&
             gap.missed_frames == 0 && gap.phase_nis == 0.0 &&
             gap.time_since_measurement_s == 0.0 &&
             std::abs(gap.future_phase_rad - gap.phase_rad) < 1e-12,
         "oversized dt must reset and cold-start from the current measurement");
  const double initial_phase_variance = std::pow(
      config.phase_kf_initial_phase_std_deg * CV_PI / 180.0, 2);
  const double initial_speed_variance = std::pow(
      config.phase_kf_initial_speed_std_deg_s * CV_PI / 180.0, 2);
  expect(std::abs(gap.phase_variance_rad2 - initial_phase_variance) < 1e-12 &&
             std::abs(gap.speed_variance_rad2_s2 -
                      initial_speed_variance) < 1e-12,
         "dt reset must restore the configured initial covariance");

  const auto backwards = solver.update({}, 0.29, true);
  expect(backwards.timestamp_gap_reset &&
             backwards.track_state == TrackState::kLost &&
             !backwards.model_valid &&
             backwards.phase_variance_rad2 == 0.0 &&
             backwards.speed_variance_rad2_s2 == 0.0,
         "non-monotonic timestamp must invalidate the filter");
}

void test_phase_nis_gate_is_preupdate() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;
  config.track_confirm_hits = 1;
  config.max_observation_distance_px = 1000.0;
  config.phase_kf_initial_phase_std_deg = 2.0;
  config.phase_kf_initial_speed_std_deg_s = 1.0;
  config.phase_kf_measurement_std_deg = 1.0;
  config.phase_kf_accel_noise_std_deg_s2 = 1.0;
  config.phase_nis_gate = 9.0;
  config.resolve_prediction_horizon(30.0);

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 0};
  geometry.axis_sin = {0, 24};
  const auto samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape =
      calibrate_phase_quad_model(samples, geometry, config);
  MotionPrior prior;
  RigidArmorSolver solver(config, geometry, shape, prior);

  const auto initial = solver.update({samples[0]}, 0.0, true);
  const auto rejected =
      solver.update({samples[6]}, 1.0 / 30.0, true);
  expect(rejected.candidate_available &&
             !rejected.measurement_used &&
             !rejected.candidate_measurement_used &&
             !rejected.phase_gate_passed &&
             rejected.phase_nis > config.phase_nis_gate &&
             rejected.phase_innovation_variance > 0.0 &&
             rejected.association_cost >= rejected.phase_nis &&
             rejected.track_state == TrackState::kTempLost &&
             rejected.missed_frames == 1,
         "large pre-update NIS must reject assimilation but keep the candidate visible");
  const double recomputed_nis =
      rejected.phase_innovation_rad * rejected.phase_innovation_rad /
      rejected.phase_innovation_variance;
  expect(std::abs(rejected.phase_nis - recomputed_nis) < 1e-9,
         "reported NIS must use the pre-update innovation covariance");
  expect(std::abs(rejected.phase_rad - initial.phase_rad) < 1e-12,
         "a rejected innovation must not pull the KF state");

  const auto recovered =
      solver.update({samples[1]}, 2.0 / 30.0, true);
  expect(recovered.measurement_used && recovered.phase_gate_passed &&
             recovered.phase_nis < config.phase_nis_gate &&
             recovered.track_state == TrackState::kTracking &&
             recovered.missed_frames == 0 &&
             recovered.time_since_measurement_s == 0.0 &&
             recovered.phase_variance_rad2 < rejected.phase_variance_rad2,
         "an in-gate measurement must recover tracking and reduce covariance");
}

void test_geometry_calibration_rejects_duplicate_centers() {
  Config config;
  config.min_geometry_samples = 5;
  config.ellipse_ransac_iterations = 5;
  ArmorObservation repeated;
  repeated.center = {100.0F, 90.0F};
  repeated.size = {20.0F, 10.0F};
  repeated.corners = {{{90.0F, 85.0F}, {110.0F, 85.0F},
                       {110.0F, 95.0F}, {90.0F, 95.0F}}};
  const std::vector<ArmorObservation> samples(10, repeated);
  expect_throws(
      [&] { calibrate_affine_geometry(samples, config, {668, 688}); },
      "RANSAC must reject fewer than five distinct centers without looping");
}

void test_calibration_profile_roundtrip_and_version_guard() {
  Config config;
  config.shape_phase_bins = 36;
  config.shape_min_samples_per_bin = 1;
  config.shape_min_covered_bins = 12;
  config.shape_smoothing_radius_bins = 2;

  AffineGeometry geometry;
  geometry.center = {100, 90};
  geometry.axis_cos = {42, 3};
  geometry.axis_sin = {-2, 24};
  geometry.calibration_samples = 72;
  geometry.calibration_rms_px = 1.25;
  const auto samples = synthetic_phase_samples(geometry);
  const PhaseQuadModel shape =
      calibrate_phase_quad_model(samples, geometry, config);

  CalibrationProfile profile;
  profile.image_size = {668, 688};
  profile.undistorted = true;
  profile.camera_reference_size = {668, 688};
  profile.camera_matrix = {800.0, 0.0, 334.0,
                           0.0, 810.0, 344.0,
                           0.0, 0.0, 1.0};
  profile.distortion = {-0.10, 0.02, 0.001, -0.002, 0.003};
  profile.sample_count = static_cast<int>(samples.size());
  profile.geometry = geometry;
  profile.phase_quad_model = shape;
  TemporaryConfig temporary;
  temporary.path = std::filesystem::temp_directory_path() /
                   "outpost_calibration_profile_test.yaml";
  save_calibration_profile(temporary.path.string(), profile);
  const CalibrationProfile loaded =
      load_calibration_profile(temporary.path.string());

  expect(loaded.version == CalibrationProfile::kCurrentVersion &&
             loaded.image_size == profile.image_size &&
             loaded.undistorted == profile.undistorted &&
             loaded.sample_count == profile.sample_count &&
             loaded.camera_reference_size == profile.camera_reference_size &&
             cv::norm(loaded.camera_matrix - profile.camera_matrix) < 1e-12 &&
             cv::norm(loaded.distortion - profile.distortion) < 1e-12,
         "calibration profile metadata must survive a YAML round trip");
  expect(cv::norm(loaded.geometry.center - profile.geometry.center) < 1e-12 &&
             cv::norm(loaded.geometry.axis_cos - profile.geometry.axis_cos) < 1e-12 &&
             cv::norm(loaded.geometry.axis_sin - profile.geometry.axis_sin) < 1e-12 &&
             loaded.phase_quad_model.bin_count() == shape.bin_count() &&
             loaded.phase_quad_model.covered_bins() == shape.covered_bins(),
         "calibration geometry and phase-bin coverage must survive serialization");
  for (double phase : {0.0, 0.7, CV_PI, kTwoPi - 0.1}) {
    const auto expected = shape.project(2, phase, geometry, {});
    const auto actual = loaded.phase_quad_model.project(
        2, phase, loaded.geometry, {});
    expect(expected.valid == actual.valid &&
               cv::norm(expected.center - actual.center) < 1e-6,
           "loaded phase model must reproduce calibrated projections");
    for (int corner = 0; corner < 4; ++corner)
      expect(cv::norm(expected.corners[corner] - actual.corners[corner]) < 1e-5,
             "loaded phase-model corners must match their saved values");
  }

  CalibrationProfile degenerate = profile;
  degenerate.geometry.axis_sin = degenerate.geometry.axis_cos;
  expect_throws(
      [&] { save_calibration_profile(temporary.path.string(), degenerate); },
      "degenerate calibration geometry must be rejected before serialization");

  profile.version = CalibrationProfile::kCurrentVersion + 1;
  save_calibration_profile(temporary.path.string(), profile);
  expect_throws([&] { load_calibration_profile(temporary.path.string()); },
                "unsupported calibration profile versions must be rejected");
}

}  // namespace

int main() {
  try {
    test_config_loading_and_validation();
    test_light_band_quad();
    test_single_light_is_not_full_armor();
    test_no_speed_prior_injection();
    test_robust_image_velocity_prediction();
    test_continuous_future_filter_and_render_hysteresis();
    test_periodic_handover_identity_and_future_id();
    test_learned_handover_cold_start();
    test_candidate_measurement_semantics();
    test_phase_quad_model_and_render_sources();
    test_tracker_lifecycle_uses_elapsed_time();
    test_timestamp_discontinuity_resets_filter();
    test_phase_nis_gate_is_preupdate();
    test_geometry_calibration_rejects_duplicate_centers();
    test_calibration_profile_roundtrip_and_version_guard();
    std::cout << "All outpost tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return 1;
  }
}
