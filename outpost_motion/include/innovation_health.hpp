#pragma once
#include <opencv2/core.hpp>
#include <deque>
#include <cmath>
#include <stdexcept>
namespace outpost {
struct HealthConfig {
  double window_seconds=.5, nis_warning=5.991464547, failure_fraction=.4;
  int min_samples=8;
};
struct HealthSummary {
  int samples=0;
  double mean_nis=0, failure_fraction=0;
  cv::Point2d mean_standardized_residual{};
  bool ready=false, warning=false;
};
// Advisory only: accepted, pre-update innovations are selection-biased.
// Never use this statistic as calibrated confidence or silently suppress output.
class InnovationHealth {
 public:
  void clear(){samples_.clear();}
  void add(double time,cv::Point2d residual,cv::Point2d variance,const HealthConfig& c){
    if(!std::isfinite(time)||!std::isfinite(residual.x)||!std::isfinite(residual.y)||
       !std::isfinite(variance.x)||!std::isfinite(variance.y)||variance.x<=0||variance.y<=0||
       (!samples_.empty()&&time<=samples_.back().time))
      throw std::runtime_error("invalid innovation health sample");
    samples_.push_back({time,{residual.x/std::sqrt(variance.x),residual.y/std::sqrt(variance.y)}});
    while(!samples_.empty()&&time-samples_.front().time>c.window_seconds+1e-9)samples_.pop_front();
  }
  HealthSummary summary(const HealthConfig& c)const{
    HealthSummary out;out.samples=int(samples_.size());
    if(samples_.empty())return out;
    for(const auto& s:samples_){const double nis=s.z.dot(s.z);
      out.mean_nis+=nis;out.failure_fraction+=nis>c.nis_warning;
      out.mean_standardized_residual+=s.z;}
    out.mean_nis/=out.samples;out.failure_fraction/=out.samples;
    out.mean_standardized_residual*=1./out.samples;
    out.ready=out.samples>=c.min_samples;
    out.warning=out.ready&&out.failure_fraction>=c.failure_fraction;
    return out;
  }
 private:
  struct Sample {double time;cv::Point2d z;};
  std::deque<Sample> samples_;
};
}
