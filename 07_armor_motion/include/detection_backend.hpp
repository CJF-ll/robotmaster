#pragma once
#include "detector.hpp"
#include <memory>
#include <string>
#include <stdexcept>
#include <chrono>
namespace outpost {
using DetectionClock=std::chrono::steady_clock;
inline double detection_ms(DetectionClock::time_point begin,DetectionClock::time_point end){
  return std::chrono::duration<double,std::milli>(end-begin).count();
}
struct DetectionTiming {
  double traditional_ms=0,network_preprocess_ms=0,network_infer_ms=0,network_decode_ms=0;
  double classification_ms=0,matching_ms=0;
  size_t network_candidates=0,classifier_calls=0;
};
struct DetectionFrame {
  DetectionTiming timing;
  struct Review {Detection candidate;std::string decision;};
  std::vector<Review> reviews;
  std::vector<Detection> observations;
  std::vector<Lightbar> lamps;
  std::vector<Detection> pairs;
  std::vector<Detector::PairStatus> pair_status;
  size_t border_rejected=0,thin_accepted=0;
  std::string backend;
};
// One pixel plane, TL/TR/BR/BL centerline points, display geometry separately.
// Network confidence must not be substituted for the geometric pairing score.
class DetectionBackend {
 public:
  virtual ~DetectionBackend()=default;
  virtual DetectionFrame detect(const cv::Mat& image)=0;
  virtual void reconfigure(const DetectorConfig&){throw std::runtime_error("backend cannot reconfigure");}
};
class TraditionalBackend final:public DetectionBackend {
 public:
  explicit TraditionalBackend(DetectorConfig config):detector_(config){}
  void reconfigure(const DetectorConfig& config)override{detector_=Detector(config);}
  DetectionFrame detect(const cv::Mat& image)override{
    const auto begin=DetectionClock::now();
    DetectionFrame f;f.observations=detector_.detect(image);f.lamps=detector_.bars();
    f.pairs=detector_.pair_candidates();f.pair_status=detector_.pair_statuses();
    f.border_rejected=detector_.border_rejected_contours();f.thin_accepted=detector_.thin_accepted_lamps();
    f.backend="traditional";f.timing.traditional_ms=detection_ms(begin,DetectionClock::now());return f;
  }
 private:Detector detector_;
};
std::unique_ptr<DetectionBackend> make_detector(const std::string& backend,DetectorConfig c);
}
