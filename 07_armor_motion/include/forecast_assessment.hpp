#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
namespace outpost {
enum ForecastBlock : unsigned {
  ForecastHistory=1, ForecastDeadline=2, ForecastUncertainty=4, ForecastGeometry=8,
  ForecastIdentity=16, ForecastShape=32
};
struct ForecastAssessment {
  unsigned blocks=0;
  double position_std=0;
  bool usable()const{return blocks==0;}
};
// Model-conditional validity only, not calibrated accuracy or render permission.
// identity_confirmed refers to local association ownership, not physical plate ID.
inline ForecastAssessment assess_forecast(bool history_ready,bool geometry_valid,
    const std::array<double,2>& variance,double age,double horizon,double deadline,double max_std,
    bool identity_confirmed=true){
  ForecastAssessment result;
  if(!history_ready)result.blocks|=ForecastHistory;
  if(!geometry_valid)result.blocks|=ForecastGeometry;
  if(!identity_confirmed)result.blocks|=ForecastIdentity;
  if(!std::isfinite(age)||!std::isfinite(horizon)||!std::isfinite(deadline)||
     age<0||horizon<0||deadline<0||age+horizon>deadline)result.blocks|=ForecastDeadline;
  bool covariance_valid=true;
  for(double v:variance)covariance_valid=covariance_valid&&std::isfinite(v)&&v>=0;
  result.position_std=covariance_valid?std::sqrt(std::max(variance[0],variance[1])):
                                        std::numeric_limits<double>::infinity();
  if(!covariance_valid||!std::isfinite(max_std)||max_std<=0||result.position_std>max_std)
    result.blocks|=ForecastUncertainty;
  return result;
}
inline std::string forecast_block_names(unsigned blocks){
  std::string result;
  auto append=[&](const char* name){if(!result.empty())result+='|';result+=name;};
  if(blocks&ForecastHistory)append("insufficient_history");
  if(blocks&ForecastDeadline)append("prediction_deadline");
  if(blocks&ForecastUncertainty)append("position_uncertainty");
  if(blocks&ForecastGeometry)append("invalid_geometry");
  if(blocks&ForecastIdentity)append("identity_unconfirmed");
  if(blocks&ForecastShape)append("shape_change_unreliable");
  return result.empty()?"none":result;
}
}
