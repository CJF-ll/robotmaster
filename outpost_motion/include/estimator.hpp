#pragma once
#include "core.hpp"
#include "target_model.hpp"
#include <optional>

namespace outpost {
// Shared runtime/evaluation boundary. Image-space and metric estimates remain
// different types; a caller must not silently interpret one as the other.
struct EstimatorOutput {
  TargetEstimate image;
  std::optional<RigidEstimate> rigid;
  int epoch=0;
};
class Estimator {
 public:
  virtual ~Estimator()=default;
  virtual EstimatorOutput update(double time,const std::vector<Detection>&,double horizon)=0;
  virtual void reconfigure(const Config&,cv::Size)=0;
  virtual const char* mode()const=0;
};
std::unique_ptr<Estimator> make_estimator(const Config&,cv::Size);
// Call before creating evaluation files or opening the source. A tool that
// scores local image tracks cannot stand in for calibrated slot evaluation.
void require_estimator_mode(const Config&,const std::string& supported,const std::string& tool);
}
