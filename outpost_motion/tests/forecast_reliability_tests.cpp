#include "tracker.hpp"
#include <iostream>
#include <stdexcept>
#include <map>
using namespace outpost;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
Detection observation(float x){
  Detection d;d.center={x,50};d.corners={cv::Point2f(x-10,44),{x+10,44},{x+10,56},{x-10,56}};
  d.measurement_points=d.corners;return d;
}
double truth(double t){
  if(t<1)return 100;
  if(t<2)return 100+60*(t-1);
  if(t<3)return 160;
  if(t<4)return 160-60*(t-3);
  if(t<5)return 100+60*(t-4);
  return 160;
}
int main(){try{
  check(assess_forecast(true,true,{4,9},0,.1,.3,20).usable(),"valid forecast rejected");
  check(assess_forecast(false,true,{4,9},0,.1,.3,20).blocks==ForecastHistory,"history reason missing");
  check(assess_forecast(true,true,{4,9},.25,.1,.3,20).blocks==ForecastDeadline,"deadline reason missing");
  check(assess_forecast(true,true,{4,900},0,.1,.3,20).blocks==ForecastUncertainty,"uncertainty reason missing");
  check(assess_forecast(true,false,{4,9},0,.1,.3,20).blocks==ForecastGeometry,"geometry reason missing");
  check(assess_forecast(false,false,{4,NAN},.25,.1,.3,20).blocks==15,"simultaneous reasons lost");
  for(double bad:{-1.,double(INFINITY),double(NAN)})
    check(!assess_forecast(true,true,{4,bad},0,.1,.3,20).usable(),"invalid covariance accepted");
  check(forecast_block_names(15)=="insufficient_history|prediction_deadline|position_uncertainty|invalid_geometry",
        "reason labels mismatch");
  TrackerConfig config;Tracker model(config);int scored=0,steady_scored=0;double sum=0,max_error=0,steady_max=0;
  struct ErrorGroup {int count=0;double sum=0,maximum=0;};std::map<std::string,ErrorGroup> groups;
  for(int i=0;i<180;++i){const double time=i/30.;
    const auto tracks=model.step(time,{observation(float(truth(time)))},.1);
    if(i<config.confirmation_hits-1)continue;
    check(tracks.size()==1&&tracks[0].id==1&&tracks[0].measurement_used,"start/stop/reversal fragmented identity");
    const auto& track=tracks[0];
    check(track.future_valid==track.forecast.usable(),"status diverged from prediction validity");
    if(!track.future_valid){check(track.forecast.blocks==ForecastHistory,"unexpected synthetic suppression");continue;}
    const double error=std::abs(track.future_center.x-truth(time+.1));
    // The switch sample has the same position under both possible future
    // motions. Only the following measurement can reveal the velocity change.
    const int phase=i%30;
    const std::string group=i>=30&&phase==0?"switch_no_new_velocity_evidence":
      i>=30&&phase<=11?"after_first_changed_measurement":
      i<150&&phase>=28?"forecast_crosses_unseen_switch":"other";
    auto& g=groups[group];++g.count;g.sum+=error;g.maximum=std::max(g.maximum,error);
    ++scored;sum+=error;max_error=std::max(max_error,error);
    if(i%30>=12&&i%30<=26){++steady_scored;steady_max=std::max(steady_max,error);}
  }
  check(scored==180-config.prediction_hits+1,"maneuver errors hidden by withholding forecasts");
  check(steady_scored==90&&steady_max<1.,"forecast failed to settle after maneuver");
  check(groups["switch_no_new_velocity_evidence"].count==5&&
        groups["after_first_changed_measurement"].count==55&&
        groups["forecast_crosses_unseen_switch"].count==10,"maneuver phase accounting wrong");
  int accounted=0;for(const auto& entry:groups)accounted+=entry.second.count;
  check(accounted==scored,"phase report omitted forecasts");
  // Missing measurements propagate state/uncertainty; recovery must reconfirm.
  Tracker coast(config);TrackOutput before;
  for(int i=0;i<30;++i){auto r=coast.step(i/30.,{observation(float(100+40*i/30.))},.1);if(!r.empty())before=r[0];}
  double uncertainty=before.forecast.position_std;
  for(int i=30;i<=32;++i){const auto r=coast.step(i/30.,{},.1);
    check(r.size()==1&&r[0].id==1&&!r[0].measurement_used,"short coast lost identity");
    check(r[0].forecast.position_std>uncertainty,"coast uncertainty did not grow");uncertainty=r[0].forecast.position_std;
    check(r[0].future_center.x>before.future_center.x,"coast position froze");}
  for(int i=33;i<38;++i){const auto r=coast.step(i/30.,{observation(float(100+40*i/30.))},.1);
    check(r.size()==1&&r[0].id==1&&r[0].measurement_used,"recovery lost identity");
    check(r[0].forecast.blocks==(i<37?unsigned(ForecastHistory):0u),"recovery reason/confirmation wrong");}
  const auto expired_horizon=coast.step(38/30.,{},.4);
  check(expired_horizon.size()==1&&(expired_horizon[0].forecast.blocks&ForecastDeadline),"long horizon lacked deadline reason");
  check(coast.step(2.,{},.1).empty(),"expired model remained active");
  std::cout<<"synthetic start/stop/reversal: scored="<<scored<<" mean_error_px="<<sum/scored
           <<" max_error_px="<<max_error<<" steady_scored="<<steady_scored<<" steady_max_px="<<steady_max<<'\n';
  for(const auto& entry:groups)std::cout<<entry.first<<" scored="<<entry.second.count
    <<" mean_error_px="<<entry.second.sum/entry.second.count<<" max_error_px="<<entry.second.maximum<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
