#include "tracker.hpp"
#include <stdexcept>
#include <iostream>
#include <limits>
using namespace outpost;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Detection observation(float x){
  Detection d;d.center={x,50};d.corners={cv::Point2f(x-10,44),{x+10,44},{x+10,56},{x-10,56}};
  d.measurement_points=d.corners;return d;
}
int main(){try{
  check(assess_forecast(true,true,{4,4},0,.1,.3,20,false).blocks==ForecastIdentity,
        "identity ambiguity absent from assessment");
  check(forecast_block_names(ForecastIdentity)=="identity_unconfirmed","identity label missing");
  const auto all_blocked=assess_forecast(false,false,
    {std::numeric_limits<double>::quiet_NaN(),4},.25,.1,.3,20,false);
  check(all_blocked.blocks==31&&!all_blocked.usable(),"combined forecast reasons lost a block");
  check(forecast_block_names(all_blocked.blocks)==
        "insufficient_history|prediction_deadline|position_uncertainty|invalid_geometry|identity_unconfirmed",
        "combined forecast reason names incomplete");
  TrackerConfig config;Tracker ambiguous(config),empty(config);
  for(int i=0;i<10;++i){ambiguous.step(i/30.,{observation(100),observation(108)},.1);
    empty.step(i/30.,{observation(100),observation(108)},.1);}
  const auto disputed=ambiguous.step(10/30.,{observation(104)},.1);
  const auto coasting=empty.step(10/30.,{},.1);
  check(disputed.size()==2&&coasting.size()==2,"ambiguity created or removed tracks");
  for(size_t i=0;i<2;++i){const auto& a=disputed[i];const auto& b=coasting[i];
    check(a.id==b.id&&!a.measurement_used&&!b.measurement_used,"ambiguous measurement updated ownership");
    check(a.filtered_center==b.filtered_center&&a.velocity==b.velocity&&a.current==b.current&&
          a.future==b.future&&a.future_position_variance==b.future_position_variance,
          "identity guard altered coasting motion or geometry");
    check(!a.future_valid&&(a.forecast.blocks&ForecastIdentity),"ambiguous forecast remained usable");
    check(b.future_valid&&!(b.forecast.blocks&ForecastIdentity),"ordinary missing observation became ambiguity");
  }
  const auto still_disputed=ambiguous.step(11/30.,{},.1);
  for(const auto& t:still_disputed)check(!t.future_valid&&(t.forecast.blocks&ForecastIdentity),
                                       "empty frame erased identity uncertainty");
  for(int i=12;i<17;++i){const auto out=ambiguous.step(i/30.,{observation(100),observation(108)},.1);
    check(out.size()==2,"recovery lost existing IDs");
    for(const auto& t:out){check(t.measurement_used&&t.id>=1&&t.id<=2,"recovery failed association");
      if(i<16)check(!t.future_valid&&(t.forecast.blocks&ForecastIdentity),"identity recovered without required history");
      else check(t.future_valid&&t.forecast.blocks==0,"confirmed recovery remained blocked");}
  }
  // Another interruption must restart recovery, and expiry must release it.
  ambiguous.step(17/30.,{observation(104)},.1);
  ambiguous.step(18/30.,{observation(100),observation(108)},.1);
  ambiguous.step(19/30.,{},.1);
  const auto renewed=ambiguous.step(20/30.,{observation(100),observation(108)},.1);
  for(const auto& t:renewed)check(t.consecutive_hits==1&&!t.future_valid&&(t.forecast.blocks&ForecastIdentity),
                                "interrupted recovery retained confirmation credit");
  check(ambiguous.step(2.,{},.1).empty(),"ambiguous identity extended track lifetime");
  std::vector<TrackOutput> fresh;
  for(int i=0;i<5;++i)fresh=ambiguous.step(2.1+i/30.,{observation(100)},.1);
  check(fresh.size()==1&&fresh[0].id==3&&fresh[0].future_valid&&fresh[0].forecast.blocks==0,
        "expired ambiguity leaked into a new identity");
  Tracker isolated(config);
  for(int i=0;i<10;++i)isolated.step(i/30.,{observation(100),observation(108),observation(300)},.1);
  const auto mixed=isolated.step(10/30.,{observation(104),observation(300)},.1);
  check(mixed.size()==3,"local ambiguity changed unrelated track count");
  for(const auto& t:mixed){
    if(t.id==3)check(t.measurement_used&&t.future_valid&&t.forecast.blocks==0,
                    "local ambiguity blocked an unrelated accepted track");
    else check(!t.measurement_used&&!t.future_valid&&(t.forecast.blocks&ForecastIdentity),
               "unrelated accepted track cleared ambiguous ownership");
  }
  std::cout<<"ambiguous ownership blocks model forecasts without changing motion; confirmed recovery restores them\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
