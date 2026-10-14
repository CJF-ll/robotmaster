#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
namespace outpost {
struct CovarianceSample {double time;cv::Vec4d value;cv::Matx44d covariance;};
struct CovarianceFit {
  cv::Vec4d position{},velocity{};
  cv::Matx44d position_covariance{},velocity_covariance{},cross_covariance{};
};
// Full 4x4 observation covariance, including cross-coordinate correlations.
// Coordinates may be metres/radians; the covariance carries the same units.
inline CovarianceFit covariance_line_fit(const std::vector<CovarianceSample>& samples,double anchor,double robust_gate=3.){
  if(!std::isfinite(robust_gate)||robust_gate<=0)throw std::runtime_error("invalid robust fit gate");
  if(samples.size()<2||samples.back().time-samples.front().time<=1e-9)
    throw std::runtime_error("insufficient covariance fit time support");
  using M8=cv::Matx<double,8,8>;using V8=cv::Vec<double,8>;using H48=cv::Matx<double,4,8>;
  M8 normal{},inverse{};V8 solution{};std::vector<double> robust(samples.size(),1.);
  std::vector<cv::Matx44d> information;
  for(const auto& s:samples){cv::Mat eigen;
    if(!cv::checkRange(cv::Mat(s.covariance))||cv::norm(cv::Mat(s.covariance-s.covariance.t()))>1e-8||
       !cv::eigen(cv::Mat(s.covariance),eigen)||eigen.at<double>(3)<=0)
      throw std::runtime_error("nonpositive observation covariance");
    information.push_back(s.covariance.inv(cv::DECOMP_CHOLESKY));}
  auto design=[&](double time){H48 h{};for(int k=0;k<4;++k){h(k,k)=1;h(k,k+4)=time-anchor;}return h;};
  for(int iteration=0;iteration<4;++iteration){normal=M8();V8 rhs{};
    for(size_t i=0;i<samples.size();++i){const auto h=design(samples[i].time);const auto w=information[i]*robust[i];
      normal+=h.t()*w*h;rhs+=h.t()*w*samples[i].value;}
    if(!cv::solve(normal,rhs,solution,cv::DECOMP_CHOLESKY))throw std::runtime_error("singular covariance fit");
    if(iteration==3)break;
    for(size_t i=0;i<samples.size();++i){const auto r=samples[i].value-design(samples[i].time)*solution;
      const double distance=std::sqrt(std::max(0.,r.dot(information[i]*r)));
      robust[i]=std::min(1.,robust_gate/std::max(1e-12,distance));}
  }
  inverse=normal.inv(cv::DECOMP_CHOLESKY);cv::Matx44d scatter{};
  for(const auto& s:samples){const auto r=s.value-design(s.time)*solution;scatter+=r*r.t();}
  scatter*=1./std::max(1.,double(samples.size())-2);M8 covariance{};
  for(size_t i=0;i<samples.size();++i){const auto gain=inverse*design(samples[i].time).t()*information[i]*robust[i];
    covariance+=gain*(samples[i].covariance+scatter)*gain.t();}
  if(!cv::checkRange(cv::Mat(solution))||!cv::checkRange(cv::Mat(covariance)))throw std::runtime_error("nonfinite covariance fit");
  CovarianceFit result;
  for(int k=0;k<4;++k){result.position[k]=solution[k];result.velocity[k]=solution[k+4];
    for(int j=0;j<4;++j){result.position_covariance(k,j)=covariance(k,j);
      result.velocity_covariance(k,j)=covariance(k+4,j+4);result.cross_covariance(k,j)=covariance(k,j+4);}}
  return result;
}
}
