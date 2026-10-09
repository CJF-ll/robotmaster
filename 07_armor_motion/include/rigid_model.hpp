#pragma once

#include <deque>
#include <optional>
#include <vector>

#include "config.hpp"
#include "types.hpp"

struct AffineGeometry {
  cv::Point2d center{};
  cv::Point2d axis_cos{};
  cv::Point2d axis_sin{};
  double armor_width_scale = 0.26;
  double armor_height_px = 18.0;
  int calibration_samples = 0;
  double calibration_rms_px = 0.0;

  cv::Point2d point(double phase) const;
  cv::Point2d tangent(double phase) const;
  cv::Point2d normalized_coordinates(const cv::Point2d& point) const;
  double phase_of(const cv::Point2d& point) const;
};

struct MotionPrior {
  bool valid = false;
  double speed_abs_rad_s = 0.0;
  int phase_direction_sign = 0;
  int repeat_period_frames = 0;
  int detected_motion_start_frame = -1;
  int detected_motion_end_frame = -1;
  double calibration_fps = 30.0;

  struct HandoverSlotPrior {
    bool valid = false;
    int transition_samples = 0;
    ArmorObservation exit_template{};
    ArmorObservation entry_template{};
    double progress_age_weight_px = 0.0;
    double progress_threshold = 0.0;
    int age_threshold_frames = 0;
  };
  struct HandoverPrior {
    bool valid = false;
    cv::Point2d forward_direction_normalized{};
    cv::Point2d forward_direction_image{};
    int first_transition_frame = -1;
    int last_active_frame = -1;
    std::array<HandoverSlotPrior, 3> slots{};
  } handover;
};

AffineGeometry calibrate_affine_geometry(const std::vector<ArmorObservation>& samples,
                                         const Config& config, cv::Size image_size);

class PhaseQuadModel {
 public:
  bool valid() const { return valid_; }
  int bin_count() const { return static_cast<int>(bins_.size()); }
  int covered_bins() const { return covered_bins_; }
  ProjectedArmor project(int id, double phase, const AffineGeometry& geometry,
                         const cv::Point2d& model_offset) const;

 private:
  struct Bin {
    cv::Point2d center_residual{};
    std::array<cv::Point2d, 4> corner_offsets{};
  };
  bool valid_ = false;
  int covered_bins_ = 0;
  std::vector<Bin> bins_;

  friend PhaseQuadModel calibrate_phase_quad_model(
      const std::vector<ArmorObservation>&, const AffineGeometry&, const Config&);
};

PhaseQuadModel calibrate_phase_quad_model(const std::vector<ArmorObservation>& samples,
                                          const AffineGeometry& geometry,
                                          const Config& config);

class RigidArmorSolver {
 public:
  RigidArmorSolver(const Config& config, AffineGeometry geometry,
                   PhaseQuadModel phase_quad_model, MotionPrior motion_prior);
  SolverOutput update(const std::vector<ArmorObservation>& observations, double timestamp_s,
                      bool scene_moving);

 private:
  Config config_;
  AffineGeometry geometry_;
  PhaseQuadModel phase_quad_model_;
  MotionPrior motion_prior_;
  cv::Point2d model_offset_{};
  bool initialized_ = false;
  double last_timestamp_s_ = 0.0;
  double phase_ = 0.0, speed_ = 0.0, acceleration_ = 0.0;
  double previous_speed_ = 0.0;
  int missed_frames_ = 0;
  MotionMode committed_mode_ = MotionMode::kInitializing;
  MotionMode pending_mode_ = MotionMode::kInitializing;
  int pending_mode_frames_ = 0;
  bool scene_moving_ = false;
  bool detection_history_valid_ = false;
  ArmorObservation previous_detection_{};
  double previous_detection_time_s_ = 0.0;
  cv::Point2d detection_velocity_px_s_{};
  bool detection_velocity_valid_ = false;
  struct DetectionSample {
    double time_s = 0.0;
    cv::Point2d center{};
  };
  std::deque<DetectionSample> detection_history_;
  int display_candidate_slot_ = -1;
  int frames_since_handover_ = 0;
  struct JumpTransition {
    ArmorObservation exit;
    ArmorObservation entry;
    int slot_step = 1;
  };
  std::deque<JumpTransition> jump_transitions_;
  bool handover_direction_valid_ = false;
  cv::Point2d forward_handover_direction_{};
  struct PhaseSample { double time = 0.0; double phase = 0.0; };
  std::deque<PhaseSample> phase_samples_;
  bool future_target_filter_valid_ = false;
  int future_target_filter_slot_ = -1;
  double future_target_filter_time_s_ = 0.0;
  cv::Point2d future_target_filter_center_{};
  cv::Point2d future_target_filter_velocity_{};
  std::array<cv::Point2d, 4> future_target_filter_corner_offsets_{};

  std::array<ProjectedArmor, 3> project_all(double base_phase) const;
  ArmorSlotOutput filter_future_target(const ArmorSlotOutput& raw_target,
                                       int target_slot, double timestamp_s,
                                       SolverOutput* diagnostics);
  MotionMode classify_mode(double dt);
  double robust_phase_slope() const;
};
