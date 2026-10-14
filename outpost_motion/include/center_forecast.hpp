#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <optional>
#include <stdexcept>
namespace outpost {
struct CenterForecastConfig {
  bool enabled=false;
  int samples=9;
  double min_span_seconds=.20,max_gap_seconds=.10,blend=.5;
};
inline bool center_forecast_config_valid(const CenterForecastConfig& c){
  return c.samples>=5&&c.samples<=60&&std::isfinite(c.min_span_seconds)&&c.min_span_seconds>0&&
    std::isfinite(c.max_gap_seconds)&&c.max_gap_seconds>0&&
    std::isfinite(c.blend)&&c.blend>0&&c.blend<=1;
}
struct CenterForecast {
  cv::Point2d center;
  std::array<double,2> variance;
};
// Accepted, confirmed observations only. A separate forecast model, never an
// association measurement or an independently extrapolated set of corners.
class CenterForecastHistory {
  struct Sample {double time;cv::Point2d center;std::array<double,2> variance;};
  std::deque<Sample> samples_;
 public:
  void clear(){samples_.clear();}
  void add(double time,cv::Point2d center,std::array<double,2> variance,
           const CenterForecastConfig& c){
    if(!center_forecast_config_valid(c)||!std::isfinite(time)||
       !std::isfinite(center.x)||!std::isfinite(center.y)||
       !std::isfinite(variance[0])||!std::isfinite(variance[1])||variance[0]<=0||variance[1]<=0)
      throw std::runtime_error("invalid center forecast observation/configuration");
    if(!samples_.empty()){
      if(time<=samples_.back().time)throw std::runtime_error("nonincreasing center forecast time");
      if(time-samples_.back().time>c.max_gap_seconds+1e-9)clear();
    }
    samples_.push_back({time,center,variance});
    while(samples_.size()>size_t(c.samples))samples_.pop_front();
  }
  std::optional<CenterForecast> predict(double horizon,const CenterForecastConfig& c)const{
    if(!center_forecast_config_valid(c)||!std::isfinite(horizon)||horizon<0||
       samples_.size()!=size_t(c.samples)||
       samples_.back().time-samples_.front().time<c.min_span_seconds-1e-9)return {};
    double mean_t=0;cv::Point2d mean{};
    for(const auto& s:samples_){mean_t+=(s.time-samples_.back().time)/samples_.size();mean+=s.center*(1./samples_.size());}
    double tt=0;cv::Point2d tv{};
    for(const auto& s:samples_){const double t=s.time-samples_.back().time-mean_t;
      tt+=t*t;tv+=(s.center-mean)*t;}
    if(tt<=1e-12)return {};
    const auto velocity=tv*(1/tt);
    CenterForecast result{samples_.back().center+velocity*horizon,{0,0}};
    std::array<double,2> residual{};
    // Current measurement and fitted slope are correlated. Propagate the full
    // linear estimator weights, not an independence approximation.
    for(size_t i=0;i<samples_.size();++i){const auto& s=samples_[i];
      const double t=s.time-samples_.back().time-mean_t;
      const double weight=(i+1==samples_.size()?1.:0.)+horizon*t/tt;
      const auto e=s.center-mean-velocity*t;
      for(int a=0;a<2;++a){result.variance[a]+=weight*weight*s.variance[a];
        residual[a]+=(a?e.y:e.x)*(a?e.y:e.x);}
    }
    const double leverage=1+2*horizon*(-mean_t)/tt+horizon*horizon/tt;
    for(int a=0;a<2;++a)result.variance[a]=std::max(result.variance[a],
      leverage*residual[a]/(samples_.size()-2));
    if(!std::isfinite(result.center.x)||!std::isfinite(result.center.y)||
       !std::isfinite(result.variance[0])||!std::isfinite(result.variance[1]))return {};
    return result;
  }
};
inline CenterForecast blend_center_forecasts(const CenterForecast& base,const CenterForecast& candidate,double weight){
  CenterForecast result{base.center*(1-weight)+candidate.center*weight,{}};
  const auto difference=candidate.center-base.center;
  for(int a=0;a<2;++a){
    // Both models share measurements: use a fully correlated upper bound and
    // retain the base uncertainty floor plus a model-disagreement term.
    const double sigma=(1-weight)*std::sqrt(base.variance[a])+weight*std::sqrt(candidate.variance[a]);
    const double delta=a?difference.y:difference.x;
    result.variance[a]=std::max(base.variance[a],sigma*sigma)+weight*(1-weight)*delta*delta;
  }
  return result;
}
}
