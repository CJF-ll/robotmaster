#pragma once
#include "detector.hpp"
#include "innovation_health.hpp"
#include "forecast_assessment.hpp"
#include "center_forecast.hpp"
#include "coaxial_phase.hpp"
#include <cstdint>
#include <deque>
namespace outpost {
struct TrackerConfig {
  CoaxialConfig coaxial;
  CenterForecastConfig center_forecast;
  bool shape_guard_enabled=false;
  double max_shape_drift_px=3.;
  int shape_history_samples=4;
  // Experimental temporal quality gate; not proof of physical occlusion.
  bool one_sided_guard_enabled=false;
  double one_sided_shrink_ratio=.7,other_side_tolerance=.1;
  int one_sided_min_hits=5;
  // History can flag likely common truncation, never prove physical completeness.
  bool common_endpoint_guard_enabled=false;
  double common_endpoint_shrink_ratio=.75,common_endpoint_width_tolerance=.10;
  double common_endpoint_max_gap_seconds=.10;
  int common_endpoint_min_hits=5;
  HealthConfig health;
  double acceleration_noise = 25000, measurement_noise = 4;
  double low_resolution_noise_scale = 1;
  bool measurement_quality_enabled = false;
  double measurement_quality_max_scale = 4;
  double initial_velocity_variance = 10000;
  double association_pixels = 30, innovation_gate = 13.82;
  double max_shape_ratio = 2.5, coast_seconds = .30;
  double max_position_std = 20;
  double association_angle_degrees = 30, association_length_ratio = 2;
  double ambiguity_margin = 1;
  double association_miss_cost = 15;
  int confirmation_hits = 3, prediction_hits = 5;
  // Optional recovery of already-mature tracks after a clean, very short gap.
  // These gates never change filtering, association or missing-frame visibility.
  bool clean_recovery_enabled = false;
  double clean_recovery_max_gap_seconds = .10, clean_recovery_max_nis = 4;
  int clean_recovery_hits = 2;
};
enum class TrackState { Tentative, Tracking, Coasting };
struct TrackOutput {
  HealthSummary health;
  std::uint64_t id = 0;
  TrackState state = TrackState::Tentative;
  bool measurement_used = false, future_valid = false;
  bool future_center_valid=false, future_shape_reliable=false;
  bool center_forecast_blended=false;
  double shape_drift_px=INFINITY;
  std::array<cv::Point2f,4> current{}, future{};
  // Accepted centerline endpoints, or last measured endpoints translated by
  // the coast state. Never display padding and never a new coast measurement.
  std::array<cv::Point2f,4> lamp_support_points{};
  cv::Point2f current_center{}, future_center{};
  int detection_index=-1;
  double innovation = 0, age_seconds = 0, position_std = 0;
  bool measurement_low_resolution = false;
  double measurement_variance = 0; // zero means no current accepted measurement
  std::array<double,2> measurement_variances{};
  // Model-conditional diagnostics, not empirically calibrated confidence.
  int consecutive_hits = 0;
  bool clean_recovery_active = false, clean_recovery_used = false;
  bool endpoint_truncation_suspected=false;
  cv::Point2d filtered_center{}, velocity{};
  std::array<double,2> velocity_variance{}, future_position_variance{};
  ForecastAssessment forecast;
};
class Tracker {
 public:
  struct AssociationDiagnostic {
    std::uint64_t id;
    int detection;
    const char* reason="invalid_geometry";
    double nis=0;
    bool innovation_available=false;
  };
  const std::vector<AssociationDiagnostic>& association_diagnostics()const{return diagnostics_;}
  struct TentativeObservation {std::uint64_t id;int detection_index;};
  const std::vector<TentativeObservation>& tentative_observations()const{return tentative_;}
  explicit Tracker(TrackerConfig config):config_(config) {}
  // Runtime model changes invalidate tracks but never recycle identity IDs.
  void reconfigure(TrackerConfig config) { config_=config;tracks_.clear();diagnostics_.clear();tentative_.clear(); }
  std::vector<TrackOutput> step(double time, const std::vector<Detection>& detections,
                                double horizon);
 private:
  struct Axis {
    double position=0, velocity=0, p00=0,p01=0,p11=0;
    void predict(double dt,double q);
    void update(double value,double r);
  };
  struct Track {
    CenterForecastHistory center_history;
    InnovationHealth health;
    std::uint64_t id;
    // Motion is estimated independently of perspective corner deformation.
    std::array<Axis,2> axes;
    std::array<cv::Point2f,4> shape;
    std::array<cv::Point2f,4> measurement_shape;
    struct ShapeSample {double time;std::array<cv::Point2f,4> offsets;};
    std::deque<ShapeSample> shape_history;
    double last_observed=0;
    int hits=1, consecutive=1;
    bool confirmed=false;
    bool one_sided_guard_ready=false;
    bool common_endpoint_guard_ready=false;
    bool common_endpoint_suspect_active=false;
    bool prediction_ready=false;
    // Ambiguous ownership invalidates forecast identity until normal recovery.
    bool identity_ambiguous=false;
    // Sticky eligibility across one outage only. Any dirty/interrupted recovery
    // cancels it; missing frames never count as accepted recovery observations.
    bool clean_recovery_pending=false, clean_recovery_used=false;
  };
  TrackerConfig config_;
  std::vector<Track> tracks_;
  std::vector<AssociationDiagnostic> diagnostics_;
  std::vector<TentativeObservation> tentative_;
  std::uint64_t next_id_=1;
  double last_time_=-1;
};
}
