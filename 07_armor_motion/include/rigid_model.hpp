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
    cv::Point2d forward_direction_image{};
    std::array<HandoverSlotPrior, 3> slots{};
  } handover;
};

AffineGeometry calibrate_affine_geometry(const std::vector<ArmorObservation>& samples,
                                         const Config& config, cv::Size image_size);

struct PhaseLabeledObservation {
  ArmorObservation observation;
  double phase_rad = 0.0;
};

AffineGeometry calibrate_labeled_affine_geometry(
    const std::vector<PhaseLabeledObservation>& samples,
    const Config& config, cv::Size image_size);

class PhaseQuadModel {
 public:
  bool valid() const { return valid_; }
  int bin_count() const { return static_cast<int>(bins_.size()); }
  int covered_bins() const { return covered_bins_; }
  ProjectedArmor project(int id, double phase, const AffineGeometry& geometry,
                         const cv::Point2d& model_offset) const;
  void write(cv::FileStorage& storage) const;
  static PhaseQuadModel read(const cv::FileNode& node);

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
  friend PhaseQuadModel calibrate_labeled_phase_quad_model(
      const std::vector<PhaseLabeledObservation>&, const AffineGeometry&, const Config&);
};

PhaseQuadModel calibrate_phase_quad_model(const std::vector<ArmorObservation>& samples,
                                          const AffineGeometry& geometry,
                                          const Config& config);

PhaseQuadModel calibrate_labeled_phase_quad_model(
    const std::vector<PhaseLabeledObservation>& samples,
    const AffineGeometry& geometry, const Config& config);

struct CalibrationProfile {
  static constexpr int kCurrentVersion = 2;
  int version = kCurrentVersion;
  cv::Size image_size{};
  bool undistorted = false;
  cv::Size camera_reference_size{};
  cv::Matx33d camera_matrix = cv::Matx33d::zeros();
  cv::Vec<double, 5> distortion{};
  int sample_count = 0;
  AffineGeometry geometry;
  PhaseQuadModel phase_quad_model;
};

void save_calibration_profile(const std::string& path,
                              const CalibrationProfile& profile);
CalibrationProfile load_calibration_profile(const std::string& path);

double angular_speed_from_handover_interval(std::size_t handover_count,
                                            double interval_s);

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
  TrackState track_state_ = TrackState::kLost;
  bool filter_valid_ = false;
  int track_confirm_hits_ = 0;
  int committed_slot_ = -1;
  double last_timestamp_s_ = 0.0;
  double last_measurement_time_s_ = 0.0;
  cv::Vec2d phase_filter_state_{0.0, 0.0};
  cv::Matx22d phase_filter_covariance_ = cv::Matx22d::eye();
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
  struct HandoverEvent {
    double time_s = 0.0;
    int slot_step = 0;
  };
  std::deque<HandoverEvent> handover_events_;
  std::deque<double> physical_speed_samples_;
  bool physical_speed_valid_ = false;
  double physical_speed_ = 0.0, physical_acceleration_ = 0.0;
  std::deque<int> rotation_direction_votes_;
  bool rotation_direction_valid_ = false;
  int rotation_direction_sign_ = 0;
  bool handover_direction_valid_ = false;
  cv::Point2d forward_handover_direction_{};

  std::array<ProjectedArmor, 3> project_all(double base_phase) const;
  void reset_tracking_state();
  void initialize_phase_filter(double phase, double timestamp_s);
  void predict_phase_filter(double dt);
  void update_phase_filter(double measurement, double measurement_variance,
                           double innovation);
  double measurement_variance(const ArmorObservation& observation) const;
  MotionMode classify_mode(double dt);
};
