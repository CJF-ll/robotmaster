#pragma once
#include <opencv2/core.hpp>
#include <cmath>
#include <optional>
#include <vector>
namespace outpost {
struct TimedCenter { double time; cv::Point2f center; };
// Shadow-only image-space stationarity evidence, never a physical speed estimate.
inline bool stationary_history(const std::vector<TimedCenter>& h,double duration,double diameter){
  if(h.size()<3||!std::isfinite(duration)||duration<=0||
     !std::isfinite(diameter)||diameter<0)return false;
  for(size_t i=0;i<h.size();++i)
    if(!std::isfinite(h[i].time)||!std::isfinite(h[i].center.x)||!std::isfinite(h[i].center.y)||
       (i&&h[i].time<=h[i-1].time))return false;
  size_t first=h.size()-1;
  while(first>0&&h.back().time-h[first].time+1e-9<duration)--first;
  if(h.size()-first<3||h.back().time-h[first].time+1e-9<duration)return false;
  for(size_t i=first;i<h.size();++i)for(size_t j=i+1;j<h.size();++j)
    if(cv::norm(h[i].center-h[j].center)>diameter)return false;
  return true;
}
// Shadow-only multi-frame constant-velocity least squares; no trajectory prior.
inline std::optional<cv::Point2f> linear_forecast(
    const std::vector<TimedCenter>& history,double horizon){
  if(history.size()<5||!std::isfinite(horizon)||horizon<0)return {};
  const double end=history.back().time,span=end-history.front().time;
  if(!std::isfinite(span)||span<=1e-9)return {};
  double mt=0,mx=0,my=0;
  for(size_t i=0;i<history.size();++i){const auto& s=history[i];
    if(!std::isfinite(s.time)||!std::isfinite(s.center.x)||!std::isfinite(s.center.y)||
       (i&&s.time<=history[i-1].time))return {};
    mt+=(s.time-end)/span;mx+=s.center.x;my+=s.center.y;
  }
  mt/=history.size();mx/=history.size();my/=history.size();
  double tt=0,tx=0,ty=0;
  for(const auto& s:history){const double t=(s.time-end)/span-mt;
    tt+=t*t;tx+=t*(s.center.x-mx);ty+=t*(s.center.y-my);}
  if(tt<=1e-12)return {};
  const double dt=horizon/span-mt;
  cv::Point2f p(float(mx+dt*tx/tt),float(my+dt*ty/tt));
  if(!std::isfinite(p.x)||!std::isfinite(p.y))return {};
  return p;
}
// Diagnostic challenger only. Inputs must contain past/current accepted measurements.
// Normalized time avoids ill-conditioning with absolute timestamps.
inline std::optional<cv::Point2f> quadratic_forecast(
    const std::vector<TimedCenter>& history, double horizon) {
  if(history.size()<5||!std::isfinite(horizon)||horizon<0)return {};
  const double end=history.back().time,span=end-history.front().time;
  if(!std::isfinite(span)||span<=1e-9)return {};
  cv::Mat design(int(history.size()),3,CV_64F),values(int(history.size()),2,CV_64F);
  for(size_t i=0;i<history.size();++i){
    const auto& s=history[i];
    if(!std::isfinite(s.time)||!std::isfinite(s.center.x)||!std::isfinite(s.center.y)||
       (i&&s.time<=history[i-1].time))return {};
    const double u=(s.time-end)/span;
    design.at<double>(int(i),0)=1;design.at<double>(int(i),1)=u;
    design.at<double>(int(i),2)=u*u;
    values.at<double>(int(i),0)=s.center.x;values.at<double>(int(i),1)=s.center.y;
  }
  cv::Mat coefficients;
  if(!cv::solve(design,values,coefficients,cv::DECOMP_SVD))return {};
  const double u=horizon/span;
  cv::Point2f result;
  for(int axis=0;axis<2;++axis){
    const double value=coefficients.at<double>(0,axis)+u*coefficients.at<double>(1,axis)+
                       u*u*coefficients.at<double>(2,axis);
    if(!std::isfinite(value))return {};
    if(axis==0)result.x=float(value);else result.y=float(value);
  }
  return result;
}
}
