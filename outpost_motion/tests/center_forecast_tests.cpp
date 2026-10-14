#include "center_forecast.hpp"
#include "tracker.hpp"
#include <iostream>
#include <stdexcept>
using namespace outpost;
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
Detection measurement(float x,float y){Detection d;d.center={x,y};
  d.corners={cv::Point2f{x-10,y-10},{x+10,y-10},{x+10,y+10},{x-10,y+10}};
  d.measurement_points=d.corners;return d;}
int main(){
  CenterForecastConfig c;
  for(double fps:{15.,30.,60.}){
    c.min_span_seconds=.1;
    CenterForecastHistory h;
    for(int i=0;i<9;++i){double t=(i+.05*(i%2))/fps;
      h.add(10000+t,{100-80*t,30+20*t},{4,9},c);}
    auto p=h.predict(.1,c);const double target=8/fps+.1;
    require(p&&cv::norm(p->center-cv::Point2d(100-80*target,30+20*target))<1e-7,"irregular-time constant velocity fit");
    require(p->variance[0]>4&&p->variance[1]>9,"shared endpoint/slope covariance omitted");
    auto zero=h.predict(0,c);
    require(zero&&std::abs(zero->variance[0]-4)<1e-9,"zero horizon variance");
    require(!h.predict(-1,c)&&!h.predict(NAN,c),"invalid horizon");
    h.add(10001,{0,0},{4,4},c);require(!h.predict(.1,c),"large gap reused old history");
    h.clear();require(!h.predict(.1,c),"clear retained history");
  }
  c=CenterForecastConfig{};CenterForecastHistory h;
  for(int i=0;i<9;++i)h.add(i/30.,{100,30},{4,4},c);
  require(cv::norm(h.predict(.1,c)->center-cv::Point2d(100,30))<1e-9,"stationary target drift");
  bool rejected=false;try{h.add(8/30.,{100,30},{4,4},c);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"duplicate timestamp accepted");
  const auto mixed=blend_center_forecasts({{0,0},{4,4}},{{2,4},{9,16}},.5);
  require(cv::norm(mixed.center-cv::Point2d(1,2))<1e-9&&mixed.variance[0]>=6.25&&mixed.variance[1]>=9,"correlated fusion variance");
  TrackerConfig tc;tc.center_forecast.enabled=true;Tracker live(tc);
  TrackerConfig bc=tc;bc.center_forecast.enabled=false;Tracker base(bc);
  for(int i=0;i<18;++i){const double t=i/30.;const auto d=measurement(float(100+20*t),float(40+10*t));
    const auto a=live.step(t,{d},.1),b=base.step(t,{d},.1);
    require(a.size()==b.size(),"forecast changed confirmation");
    if(a.empty())continue;
    require(a[0].id==b[0].id&&a[0].current==b[0].current&&a[0].filtered_center==b[0].filtered_center&&
      a[0].velocity==b[0].velocity,"forecast changed association/filter/observation");
    require(a[0].center_forecast_blended==(i>=10),"forecast warmup did not use confirmed history");
    for(int k=1;k<4;++k)require(cv::norm((a[0].future[k]-a[0].future[0])-(b[0].future[k]-b[0].future[0]))<1e-4,"forecast changed silhouette");
    for(int axis=0;axis<2;++axis)require(a[0].future_position_variance[axis]>=b[0].future_position_variance[axis],"forecast reduced uncertainty floor");
  }
  const auto lost=live.step(18/30.,{},.1);
  for(const auto& t:lost)require(!t.center_forecast_blended,"missing observation produced fitted forecast");
  const auto recovered=live.step(19/30.,{measurement(float(100+20*19/30.),float(40+10*19/30.))},.1);
  for(const auto& t:recovered)require(!t.center_forecast_blended,"gap retained fit history");
  live.reconfigure(tc);
  require(live.step(20/30.,{measurement(114,47)},.1).empty(),"reconfigure reused track");
  std::cout<<"causal center forecast contracts passed\n";
}
