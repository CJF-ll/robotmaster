#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <vector>
#include <string>
namespace outpost {
struct DetectorConfig {
  std::string backend="traditional";
  std::string neural_model, classifier_model;
  double neural_score=.7, neural_nms=.3, classifier_threshold=.7;
  double classifier_crop_half_scale=1.125, support_endpoint_fraction=.5;
  int classifier_target_index=6;
  int neural_inference_threads=0; // 0: OpenVINO automatic; startup-only.
  // Independent neural-ROI experiment. Global segmentation stays unchanged.
  bool roi_refinement_enabled=false;
  int roi_brightness=90;
  double roi_padding_lengths=.6;
  // Optional original-gray endpoint measurement, independent of display padding.
  bool endpoint_refinement_enabled=false;
  bool parallel_branches=false;
  double endpoint_search_fraction=.35,endpoint_min_contrast=8;
  double endpoint_max_spread_px=1.5;
  bool thin_vertical_enabled = true;
  int thin_min_pixels = 8;
  bool reject_border_contours = true;
  int border_margin_px = 0;
  int brightness = 130, red_difference = 35;
  double min_area = 4, min_length = 5, max_length = 80;
  double min_aspect = 1.8, max_tilt_deg = 40;
  double max_length_ratio = 2, max_angle_diff_deg = 25;
  double min_separation = .6, max_separation = 4;
  double max_vertical_offset = .8;
  double end_margin_ratio = .08, side_margin_ratio = .15;
  double end_margin_min_px = 1, side_margin_min_px = .75;
  double pairing_ambiguity_margin = .05;
};
// A visible contrast transition does NOT establish physical completeness: an
// opaque occluder can produce the same single-frame edge as an intact lamp.
enum class EndpointEvidence { NotEvaluated, ContrastSupported, Uncertain, BoundaryUnsupported };
inline const char* endpoint_evidence_name(EndpointEvidence evidence){
  switch(evidence){case EndpointEvidence::NotEvaluated:return "not_evaluated";
    case EndpointEvidence::ContrastSupported:return "contrast_supported";
    case EndpointEvidence::Uncertain:return "uncertain";
    case EndpointEvidence::BoundaryUnsupported:return "boundary_unsupported";}
  return "unknown";
}
struct Lightbar {
  EndpointEvidence endpoint_evidence=EndpointEvidence::NotEvaluated;
  bool endpoint_refined=false;
  double endpoint_contrast=0,endpoint_spread_px=0;
  bool low_resolution = false;
  bool quality_available=false;
  double fit_residual_px=0;
  std::array<double,2> center_variance_scale{1,1};
  cv::Point2f center, axis;
  float length = 0, width = 0;
  std::array<cv::Point2f,4> vertices;
};
enum class ObservationKind { LampPair, NeuralKeypoints, NeuralSupportedLampPair };
inline const char* observation_kind_name(ObservationKind kind){
  switch(kind){case ObservationKind::LampPair:return "lamp_pair";
    case ObservationKind::NeuralKeypoints:return "neural_keypoints";
    case ObservationKind::NeuralSupportedLampPair:return "neural_supported_lamp_pair";}
  return "unknown";
}
struct Detection {
  std::array<EndpointEvidence,2> endpoint_evidence{
      EndpointEvidence::NotEvaluated,EndpointEvidence::NotEvaluated};
  bool endpoint_refined=false;
  double endpoint_min_contrast=0,endpoint_max_spread_px=0;
  ObservationKind kind=ObservationKind::LampPair;
  // -1 means unavailable. Never substitute these for geometric pairing score.
  double detection_confidence=-1, classification_confidence=-1;
  int raw_network_class=-1, classified_index=-1;
  bool low_resolution = false;
  // Dimensionless, geometry-derived inflation heuristics, NOT calibrated
  // covariance. Pixel residuals are measured against fitted contour boundaries.
  bool quality_available=false;
  double fit_residual_px=0;
  std::array<double,2> center_variance_scale{1,1};
  bool roi_refined=false;
  // TL TR BR BL. LampPair kinds are measured centerlines; NeuralKeypoints are
  // learned estimates, not proof of four directly visible lamp endpoints.
  std::array<cv::Point2f,4> measurement_points{};
  std::array<cv::Point2f,4> corners; // TL TR BR BL, display enclosure only
  cv::Point2f center;
  double score = 0;
  int left = -1, right = -1;
};
bool enclose(const Lightbar& left, const Lightbar& right,
             const DetectorConfig&, Detection&);
class Detector {
 public:
  explicit Detector(DetectorConfig config) : config_(config) {}
  std::vector<Detection> detect(const cv::Mat& image);
  const std::vector<Lightbar>& bars() const { return bars_; }
  size_t border_rejected_contours() const { return border_rejected_contours_; }
  size_t thin_accepted_lamps() const { return thin_accepted_lamps_; }
  enum class PairStatus { Accepted, AmbiguousLamp, SharedLamp };
  const std::vector<Detection>& pair_candidates() const { return pairs_; }
  const std::vector<PairStatus>& pair_statuses() const { return pair_statuses_; }
 private:
  DetectorConfig config_;
  size_t border_rejected_contours_ = 0;
  size_t thin_accepted_lamps_ = 0;
  std::vector<Lightbar> bars_;
  std::vector<Detection> pairs_;
  std::vector<PairStatus> pair_statuses_;
};
}
