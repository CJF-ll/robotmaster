#pragma once
#include "detector.hpp"
#include <opencv2/calib3d.hpp>
#include <array>
#include <memory>
#include <limits>
#include <string>
#include <cstdint>

namespace outpost {
struct RigidEstimatorConfig {
  bool covariance_estimation_enabled=true;
  double association_nis_scale=18.47,robust_fit_gate_std=3.;
  cv::Mat camera_matrix, distortion, camera_from_target_rotation;
  double lightbar_length_m=0, lightbar_center_distance_m=0, radius_m=0;
  double pitch_rad=std::numeric_limits<double>::quiet_NaN();
  std::array<double,3> slot_heights_m{{NAN,NAN,NAN}};
  double history_seconds=.30, min_history_seconds=.10, coast_seconds=.30;
  int min_samples=5;
  double max_reprojection_error_px=2.5, max_orientation_error_rad=.30;
  double max_center_error_m=.10, max_phase_error_rad=.65;
  double max_angular_speed_rad_s=20, max_center_speed_m_s=10;
  double min_quad_area_px=8, identity_score_margin=4, pose_ambiguity_margin=.20;
  double max_fit_center_rms_m=.03, max_fit_phase_rms_rad=.12;
  int pose_refinement_iterations=20;
  // Explicit modelling assumptions, not an empirically calibrated probability.
  double pixel_noise_std_px=.5, min_geometry_information_ratio=1e-8;
  double max_center_std_m=.15, max_phase_std_rad=.35;
  double translation_acceleration_std_m_s2=1., angular_acceleration_std_rad_s2=4.;
  double max_future_corner_std_px=20;
  bool adaptive_window_enabled=true;
  double motion_change_threshold_rad=.025, motion_change_threshold_m=.008;
  int motion_change_confirmations=3;
};
enum class RigidTrackingState { Lost, Initializing, Tracking, Coasting };
struct RigidProjectedSlot {
  int id=0, detection_index=-1;
  bool observed=false, current_valid=false, future_valid=false;
  bool future_geometry_valid=false,prediction_reliable=false;
  bool future_uncertainty_available=false;
  std::string forecast_block_reason="uninitialized";
  std::array<double,4> future_corner_std_px{{NAN,NAN,NAN,NAN}};
  double max_future_corner_std_px=NAN;
  // Geometric lamp centerline endpoints in TL/TR/BR/BL correspondence. These
  // are NOT padded detector enclosures and need not all be visible in the image.
  std::array<cv::Point2f,4> current_points{},future_points{};
};
struct RigidMeasurementDiagnostic {
  int detection_index=-1,slot=-1;
  bool accepted=false;
  double reprojection_error_px=NAN,orientation_error_rad=NAN;
  std::string reason="no_admissible_pose";
  double geometry_information_ratio=NAN,center_std_m=NAN,phase_std_rad=NAN;
};
struct RigidEstimate {
  std::string fit_method="none",fit_selection_reason="not_fitted";
  double scalar_fit_score=NAN,gls_fit_score=NAN;
  RigidTrackingState state=RigidTrackingState::Lost;
  bool initialized=false,valid=false,measurement_used=false,motion_confirmed=false;
  bool identity_resolved=false,relative_identity_only=false;
  bool association_ambiguous=false;
  bool uncertainty_available=false,state_reliable=false,adaptive_window_active=false;
  double center_std_m=NAN,phase_std_rad=NAN,omega_std_rad_s=NAN,active_history_seconds=0;
  // Increments on every acquisition, including expiry and same-frame rebirth.
  std::uint64_t generation=0;
  // In the configured target-axis basis, whose origin is the camera center.
  cv::Vec3d center{},velocity{};
  double phase=0,omega=0,time=0,age_seconds=0;
  double fit_center_rms_m=0,fit_phase_rms_rad=0;
  int sample_count=0,accepted_measurements=0;
  std::array<RigidProjectedSlot,3> slots;
  std::vector<RigidMeasurementDiagnostic> diagnostics;
};
// Online and causal. No trajectory table, fixed speed or access to future frames.
// A calibrated axis basis is required in addition to K/D and measured dimensions.
// Unequal-height cyclic identities remain unresolved until observations separate
// the three hypotheses under this motion/geometry model (not externally proven
// physical labels). Equal-height identities are relative to the first plate.
class CalibratedRigidEstimator {
 public:
  explicit CalibratedRigidEstimator(const RigidEstimatorConfig&);
  ~CalibratedRigidEstimator();
  CalibratedRigidEstimator(CalibratedRigidEstimator&&) noexcept;
  CalibratedRigidEstimator& operator=(CalibratedRigidEstimator&&) noexcept;
  CalibratedRigidEstimator(const CalibratedRigidEstimator&)=delete;
  CalibratedRigidEstimator& operator=(const CalibratedRigidEstimator&)=delete;
  RigidEstimate update(double time,const std::vector<Detection>&,double horizon);
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
