#include "tracker.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace outpost;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Detection observation(float x){
  Detection d;d.center={x,50};d.corners={cv::Point2f(x-10,44),{x+10,44},{x+10,56},{x-10,56}};
  d.measurement_points=d.corners;return d;
}
int main(){try{
  TrackerConfig config;Tracker continued(config),reversed(config);TrackOutput a,b;
  // Both worlds have identical past and current observations at t=1.
  for(int i=0;i<=30;++i){const double time=i/30.;const auto d=observation(float(160-60*time));
    const auto x=continued.step(time,{d},.1),y=reversed.step(time,{d},.1);
    check(x.size()==y.size(),"identical observation histories produced different track counts");
    if(x.empty())continue;a=x[0];b=y[0];
    check(a.id==b.id&&a.velocity==b.velocity&&a.future_center==b.future_center&&
          a.future_valid==b.future_valid,"future branch leaked into causal prediction");
  }
  check(a.future_valid&&b.future_valid,"boundary forecast silently suppressed");
  const double continued_truth=94,reversed_truth=106;
  const double e1=std::abs(a.future_center.x-continued_truth),e2=std::abs(b.future_center.x-reversed_truth);
  check(std::max(e1,e2)>=6-1e-5,"identical forecast beat two-world causal lower bound");
  // Once the first changed measurement arrives, adaptation can be measured.
  std::vector<double> velocity_errors;double post_sum=0,post_max=0;int issued=0;
  for(int i=1;i<=20;++i){const double time=1+i/30.;
    const auto x=continued.step(time,{observation(float(100-60*(time-1)))},.1);
    const auto y=reversed.step(time,{observation(float(100+60*(time-1)))},.1);
    check(x.size()==1&&y.size()==1&&x[0].id==1&&y[0].id==1&&y[0].measurement_used,
          "reversal fragmented or suppressed identity");
    check(y[0].future_valid,"reversal accuracy improved by withholding predictions");
    velocity_errors.push_back(std::abs(y[0].velocity.x-60));
    const double error=std::abs(y[0].future_center.x-(100+60*(time+.1-1)));
    post_sum+=error;post_max=std::max(post_max,error);++issued;
  }
  int settling_samples=int(velocity_errors.size())+1;
  bool suffix_good=true;
  for(int i=int(velocity_errors.size())-1;i>=0;--i){suffix_good=suffix_good&&velocity_errors[i]<=6;
    if(suffix_good)settling_samples=i+1;}
  check(settling_samples<=12,"post-observation velocity response exceeded 0.4 seconds");
  std::cout<<"identical_history horizon_seconds=0.1 continued_error_px="<<e1<<" reversed_error_px="<<e2
    <<" worst_case_lower_bound_px=6\n"
    <<"after_first_changed_measurement issued="<<issued<<" mean_error_px="<<post_sum/issued
    <<" max_error_px="<<post_max<<" velocity_within_10_percent_samples="<<settling_samples
    <<" seconds_from_switch="<<settling_samples/30.<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
