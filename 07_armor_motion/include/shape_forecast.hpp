#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <optional>
#include <vector>
#include <deque>
namespace outpost {
using Quad=std::array<cv::Point2f,4>;
struct TimedShape {double time;cv::Point2f center;Quad corners;};
struct ShapeForecastConfig {
  int history=7,evidence_samples=3,evidence_window=5;
  double trend_sigma=2,max_scale=1.8,evidence_min_gain=.1;
  bool width_only=true;
};
inline bool shape_config_valid(const ShapeForecastConfig& c){
  return c.history>=5&&c.history<=30&&std::isfinite(c.trend_sigma)&&c.trend_sigma>=0&&
    std::isfinite(c.max_scale)&&c.max_scale>=1&&c.evidence_samples>=1&&
    c.evidence_window>=c.evidence_samples&&c.evidence_window<=100&&
    std::isfinite(c.evidence_min_gain)&&c.evidence_min_gain>=0;
}
// Evidence is added only when an older forecast reaches its target frame.
struct ShapeEvidence {
  std::deque<double> gains;
  void add(double gain,const ShapeForecastConfig& c){
    if(!shape_config_valid(c)||!std::isfinite(gain))return;
    gains.push_back(gain);while(gains.size()>size_t(c.evidence_window))gains.pop_front();
  }
  bool prefer_candidate(const ShapeForecastConfig& c)const{
    if(!shape_config_valid(c)||gains.size()<size_t(c.evidence_samples))return false;
    double total=0;for(double g:gains)total+=g;return total/gains.size()>c.evidence_min_gain;
  }
};
inline bool shape_valid(const Quad& q){
  for(auto p:q)if(!std::isfinite(p.x)||!std::isfinite(p.y))return false;
  return q[0].x<q[1].x&&q[3].x<q[2].x&&q[3].y>q[0].y&&q[2].y>q[1].y&&
    cv::isContourConvex(std::vector<cv::Point2f>(q.begin(),q.end()))&&
    cv::contourArea(std::vector<cv::Point2f>(q.begin(),q.end()))>4;
}
// Diagnostic image-shape model, not a calibrated 3D pose or angular velocity.
// Full mode uses logarithmic dimensions. Width-only mode uses a linear width
// trend and rejects nonpositive/out-of-scale widths. Convexity validation rejects
// crossing sides; rejection never synthesizes a replacement rectangle.
inline std::optional<Quad> forecast_shape(const std::vector<TimedShape>& history,
    cv::Point2f future_center,double horizon,const ShapeForecastConfig& config){
  if(!shape_config_valid(config)||history.size()!=size_t(config.history)||!std::isfinite(horizon)||horizon<0||
     !std::isfinite(config.trend_sigma)||config.trend_sigma<0||
     !std::isfinite(config.max_scale)||config.max_scale<1||
     !std::isfinite(future_center.x)||!std::isfinite(future_center.y))return {};
  std::vector<std::array<double,6>> values;std::vector<double> times;
  double mean_time=0;
  for(size_t i=0;i<history.size();++i){const auto& s=history[i];
    if(!std::isfinite(s.time)||(i&&s.time<=history[i-1].time)||!shape_valid(s.corners)||
       !std::isfinite(s.center.x)||!std::isfinite(s.center.y))return {};
    const auto left=(s.corners[0]+s.corners[3])*.5f,right=(s.corners[1]+s.corners[2])*.5f;
    const auto le=s.corners[3]-s.corners[0],re=s.corners[2]-s.corners[1];
    const double width=right.x-left.x;if(width<=0)return {};
    values.push_back({config.width_only?width:std::log(width),right.y-left.y,std::log(le.y),std::log(re.y),le.x/le.y,re.x/re.y});
    times.push_back(s.time-history.back().time);mean_time+=times.back()/history.size();
  }
  double variance=0;for(double t:times)variance+=(t-mean_time)*(t-mean_time);
  if(variance<1e-12)return {};
  auto predicted=values.back();
  for(int component=0;component<6;++component){
    if(config.width_only&&component!=0)continue;
    double mean=0;for(auto v:values)mean+=v[component]/values.size();
    double covariance=0;for(size_t i=0;i<values.size();++i)covariance+=(times[i]-mean_time)*(values[i][component]-mean);
    const double slope=covariance/variance;double residual=0;
    for(size_t i=0;i<values.size();++i){const double e=values[i][component]-mean-slope*(times[i]-mean_time);residual+=e*e;}
    const double error=std::sqrt(residual/(values.size()-2)/variance);
    if(std::abs(slope)>config.trend_sigma*error)predicted[component]+=horizon*slope;
  }
  for(int component:{0,2,3}){
    if(component==0&&config.width_only){
      const double ratio=predicted[0]/values.back()[0];
      if(!std::isfinite(ratio)||ratio<1/config.max_scale||ratio>config.max_scale)return {};
    }else if(std::abs(predicted[component]-values.back()[component])>std::log(config.max_scale))return {};
  }
  const auto& last=history.back();
  const auto left=(last.corners[0]+last.corners[3])*.5f,right=(last.corners[1]+last.corners[2])*.5f;
  const auto display_center=future_center+(left+right)*.5f-last.center;
  const cv::Point2f separation(float(config.width_only?predicted[0]:std::exp(predicted[0])),float(predicted[1]));
  const float lh=float(std::exp(predicted[2])),rh=float(std::exp(predicted[3]));
  const cv::Point2f le(float(predicted[4])*lh,lh),re(float(predicted[5])*rh,rh);
  const auto r=display_center+separation*.5f,l=display_center-separation*.5f;
  Quad result={l-le*.5f,r-re*.5f,r+re*.5f,l+le*.5f};
  if(!shape_valid(result))return {};
  return result;
}
}
