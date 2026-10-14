#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <cmath>
#include <deque>
#include <optional>
#include <vector>
namespace outpost {
struct CoaxialConfig {
  bool enabled=false;
  double window_seconds=3.,fit_interval_seconds=.5;
  double min_omega=.3,max_omega=6.,omega_step=.05;
  double max_fit_rms=6.,ambiguity_ratio=1.3;
  double innovation_pixels=15.,max_phase_correction=.4;
  double coast_seconds=.3,stop_omega=.07;
  double offset_gain=.25;
  double association_pixels=30,association_margin=3;
  int min_samples=40,min_slot_samples=6,phase_samples=9,max_samples=120;
};
bool coaxial_config_valid(const CoaxialConfig&);
struct CoaxialObservation {int slot;cv::Point2d center;};
struct CoaxialState {
  int generation=0;
  bool initialized=false,valid=false,measurement_used=false;
  double phase=0,omega=0,period_seconds=0,fit_rms=0,age_seconds=0;
  // Phase sign refers to the configured slot convention, not metric CW/CCW.
  std::array<cv::Point2d,3> current{},future{};
  // Complete fitted center geometry, in pixels; never armor corner geometry.
  std::array<cv::Point2d,3> orbit_centers{};
  cv::Point2d cosine{},sine{};
  cv::Point2d model_point(int slot,double common_phase) const {
    const double a=common_phase+slot*2*CV_PI/3;
    return orbit_centers.at(slot)+cosine*std::cos(a)+sine*std::sin(a);
  }
  std::array<std::array<double,2>,3> future_variance{};
  std::vector<int> observation_slots;
  const char* reason="initializing";
};
// Online affine projection of three coaxial circles. Slot-specific centers
// represent height offsets; both sinusoidal basis vectors are shared.
// This is weak-perspective image geometry, not calibrated metric reconstruction.
class CoaxialPhaseModel {
 public:
  explicit CoaxialPhaseModel(CoaxialConfig c);
  CoaxialState update(double time,const std::vector<CoaxialObservation>&,double horizon);
 private:
  struct Sample {double time;int slot;cv::Point2d center;};
  struct Projection {
    std::array<cv::Point2d,3> centers{};
    cv::Point2d cosine{},sine{};
    double omega=0,phase=0,rms=0;
    cv::Point2d point(int slot,double common_phase)const;
    cv::Point2d derivative(int slot,double common_phase)const;
  };
  std::optional<Projection> fit(double now)const;
  CoaxialConfig config_;
  std::deque<Sample> samples_;
  std::deque<std::pair<double,double>> phases_;
  std::optional<Projection> projection_;
  double last_time_=-1,last_observed_=-1,last_fit_=-1,phase_=0,omega_=0;
  double phase_variance_=1,omega_variance_=1;
  std::array<cv::Point2d,3> offsets_{};
  int generation_=0;
};
}
