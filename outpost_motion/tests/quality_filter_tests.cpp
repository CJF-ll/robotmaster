#include "tracker.hpp"
#include <stdexcept>
#include <limits>
#include <iostream>
using namespace outpost;
void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
Detection point(float x,float y){Detection d;d.center={x,y};
  d.corners={cv::Point2f(x-10,y-6),{x+10,y-6},{x+10,y+6},{x-10,y+6}};
  d.measurement_points=d.corners;return d;}
int main(){try{
  TrackerConfig base;base.confirmation_hits=1;TrackerConfig quality=base;quality.measurement_quality_enabled=true;
  Tracker ordinary(base),weighted(quality);auto first=point(100,100);
  auto a=ordinary.step(0,{first},.1),b=weighted.step(0,{first},.1);
  check(a[0].filtered_center==b[0].filtered_center,"quality unavailable changed state");
  for(int i=1;i<12;++i){first=point(100,100);ordinary.step(i/30.,{first},.1);weighted.step(i/30.,{first},.1);}
  auto noisy=point(103,103);noisy.quality_available=true;noisy.fit_residual_px=1;
  noisy.center_variance_scale={4,1};a=ordinary.step(.4,{noisy},.1);b=weighted.step(.4,{noisy},.1);
  check(a.size()==1&&b.size()==1&&a[0].id==b[0].id,"quality broke unique association");
  check(b[0].measurement_variances[0]==16&&b[0].measurement_variances[1]==4&&b[0].measurement_variance==16,"quality variance axes lost");
  check(b[0].filtered_center.x<a[0].filtered_center.x&&b[0].filtered_center.y==a[0].filtered_center.y,"quality applied to wrong axis");
  check(b[0].current==a[0].current,"measurement weight altered real observed outline");
  Tracker birth(quality);noisy.center_variance_scale={100,2};auto born=birth.step(0,{noisy},0);
  check(born[0].future_position_variance[0]==16&&born[0].future_position_variance[1]==8,"birth variance not capped per axis");
  Tracker invalid(quality);noisy.center_variance_scale[0]=std::numeric_limits<double>::quiet_NaN();
  check(invalid.step(0,{noisy},.1).empty(),"invalid quality spawned track");
  quality.clean_recovery_enabled=true;quality.confirmation_hits=3;Tracker combined(quality);
  for(int i=0;i<12;++i)combined.step(i/30.,{point(100,100)},.1);
  combined.step(12/30.,{},.1);noisy=point(100,100);noisy.quality_available=true;noisy.center_variance_scale={2,1};
  for(int i=13;i<17;++i){auto o=combined.step(i/30.,{noisy},.1);
    check(!o[0].future_valid&&!o[0].clean_recovery_used,"inflated R bypassed clean recovery gate");}
  check(combined.step(17/30.,{noisy},.1)[0].future_valid,"normal confirmation did not recover");
  std::cout<<"quality weights association/update/birth consistently; recovery cannot exploit inflation\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
