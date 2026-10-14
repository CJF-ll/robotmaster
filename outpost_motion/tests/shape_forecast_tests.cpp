#include "shape_forecast.hpp"
#include <iostream>
#include <stdexcept>
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
int main(){
  using namespace outpost;ShapeForecastConfig config;config.width_only=false;
  std::vector<TimedShape> history;
  for(int i=0;i<7;++i){const double t=i/30.;const float width=float(30*std::exp(-t));
    const cv::Point2f c(100+float(t)*50,80);
    history.push_back({t,c,{c+cv::Point2f(-width/2,-10),c+cv::Point2f(width/2,-10),
                           c+cv::Point2f(width/2,10),c+cv::Point2f(-width/2,10)}});
  }
  const cv::Point2f future(115,80);auto result=forecast_shape(history,future,.1,config);
  require(result&&shape_valid(*result),"shrinking prediction missing or inverted");
  require(std::abs((*result)[1].x-(*result)[0].x-30*std::exp(-.3))<.001,"causal log-width trend incorrect");
  require(!forecast_shape(history,future,1,config),"excessive scale extrapolation not rejected");
  require(!forecast_shape(history,future,-.1,config),"negative horizon accepted");
  auto zero=forecast_shape(history,history.back().center,0,config);
  for(int k=0;k<4;++k)require(cv::norm((*zero)[k]-history.back().corners[k])<.001,"zero horizon changes silhouette");
  history[3].time=history[2].time;require(!forecast_shape(history,future,.1,config),"duplicate timestamps accepted");
  history[3].time=.1;std::swap(history[3].corners[0],history[3].corners[2]);
  require(!forecast_shape(history,future,.1,config),"invalid corner order accepted");
  std::cout<<"causal shape trend, positive dimensions, geometry and horizon contracts passed\n";
  ShapeEvidence evidence;require(!evidence.prefer_candidate(config),"candidate chosen without matured evidence");
  evidence.add(1,config);evidence.add(1,config);require(!evidence.prefer_candidate(config),"insufficient evidence chosen");
  evidence.add(1,config);require(evidence.prefer_candidate(config),"consistent past improvement ignored");
  for(int i=0;i<5;++i)evidence.add(-1,config);
  require(!evidence.prefer_candidate(config),"stale positive evidence overrode recent failures");
  config.width_only=true;history.clear();
  for(int i=0;i<7;++i){const double t=i/30.;const float w=30-float(t)*20;
    history.push_back({t,{100,80},{cv::Point2f(100-w/2,70),{100+w/2,70},{100+w/2,90},{100-w/2,90}}});
  }
  const auto linear=forecast_shape(history,{100,80},.1,config);
  require(linear&&std::abs((*linear)[1].x-(*linear)[0].x-24)<.001,"width-only linear trend wrong");
  require(!forecast_shape(history,{100,80},2,config),"negative width not rejected");
  for(int i=0;i<7;++i){history[i].corners=history.back().corners;}
  const auto stationary=forecast_shape(history,{100,80},.1,config);
  require(stationary&&cv::norm((*stationary)[0]-history.back().corners[0])<.001,"stationary shape deformed");
  config.evidence_window=1;require(!shape_config_valid(config),"inconsistent evidence window accepted");
}
