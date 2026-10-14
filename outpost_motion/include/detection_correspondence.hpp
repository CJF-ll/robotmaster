#pragma once
#include "detector.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
namespace outpost {
struct Correspondence {
  int traditional=-1;
  int eligible=0;
  bool ambiguous=false;
  double max_endpoint_error=0;
};
// Diagnostic only: never modifies either detector's geometry or resolves ties
// using confidence. Accept only mutually unique, four-endpoint-supported pairs.
inline std::vector<Correspondence> detection_correspondence(
    const std::vector<Detection>& neural,const std::vector<Detection>& traditional,
    double endpoint_fraction) {
  if(!std::isfinite(endpoint_fraction)||endpoint_fraction<=0||endpoint_fraction>1)
    throw std::runtime_error("endpoint_fraction must be in (0,1]");
  auto valid=[](const Detection& d){
    for(auto p:d.measurement_points)if(!std::isfinite(p.x)||!std::isfinite(p.y))return false;
    std::vector<cv::Point2f> p(d.measurement_points.begin(),d.measurement_points.end());
    return cv::isContourConvex(p)&&cv::contourArea(p)>4;
  };
  std::vector<Correspondence> result(neural.size());
  std::vector<int> degree(traditional.size(),0);
  for(size_t i=0;i<neural.size();++i){
    if(!valid(neural[i]))continue;
    for(size_t j=0;j<traditional.size();++j){
      if(!valid(traditional[j]))continue;
      const auto& p=traditional[j].measurement_points;
      const double scale=std::min(cv::norm(p[0]-p[3]),cv::norm(p[1]-p[2]));
      double error=0;for(size_t k=0;k<4;++k)
        error=std::max(error,cv::norm(neural[i].measurement_points[k]-p[k]));
      if(scale<=0||error>endpoint_fraction*scale)continue;
      ++result[i].eligible;++degree[j];
      result[i].traditional=int(j);result[i].max_endpoint_error=error;
    }
  }
  for(auto& r:result){
    r.ambiguous=r.eligible>1||(r.eligible==1&&degree[size_t(r.traditional)]>1);
    if(r.eligible!=1||r.ambiguous){r.traditional=-1;r.max_endpoint_error=0;}
  }
  return result;
}
}
