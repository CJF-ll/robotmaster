#include "tracker.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits>
using namespace outpost;
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
Detection detection(float x,float y=50){
  Detection d;d.corners={cv::Point2f(x-10,y-6),{x+10,y-6},{x+10,y+6},{x-10,y+6}};
  d.center={x,y};d.measurement_points=d.corners;return d;
}
int main(){
  {TrackerConfig c;c.confirmation_hits=1;c.prediction_hits=1;
    for(int kind=0;kind<5;++kind){Tracker guarded(c),control(c);auto bad=detection(100);
      if(kind==0)bad.center.x=NAN;
      if(kind==1)bad.center.y=INFINITY;
      if(kind==2)bad.measurement_points[0].x=NAN;
      if(kind==3){std::swap(bad.measurement_points[0],bad.measurement_points[3]);std::swap(bad.measurement_points[1],bad.measurement_points[2]);}
      if(kind==4)bad.corners[0].x=INFINITY;
      require(guarded.step(0,{bad},.1).empty(),"invalid candidate became visible");
      bool rejected=false;for(auto d:guarded.association_diagnostics())rejected|=d.id==0&&d.detection==0;
      require(rejected,"pre-birth rejection missing");control.step(0,{},.1);
      auto a=guarded.step(.04,{detection(101)},.1),b=control.step(.04,{detection(101)},.1);
      require(a.size()==1&&a[0].id==1&&cv::norm(a[0].filtered_center-b[0].filtered_center)==0,"invalid birth consumed ID/state");
      a=guarded.step(.08,{bad},.1);b=control.step(.08,{},.1);
      require(a.size()==1&&a[0].id==1&&!a[0].measurement_used&&cv::norm(a[0].filtered_center-b[0].filtered_center)==0,"invalid candidate poisoned existing track");
      a=guarded.step(.12,{bad,detection(103)},.1);
      require(a.size()==1&&a[0].id==1&&a[0].detection_index==1&&a[0].measurement_used,"rejection changed valid detection index");
    }
  }
  {TrackerConfig c;Tracker irregular(c),control(c);double time=0;std::vector<TrackOutput> a,b;
    for(int i=0;i<80;++i){time+=(i%3==0?.017:i%3==1?.041:.029);
      auto d=detection(float(100+30*time));a=irregular.step(time,{d},.1);b=control.step(time,{d},.1);
      if(i==40){for(double invalid_time:{time,time-.01,std::numeric_limits<double>::quiet_NaN()}){
        bool rejected=false;try{irregular.step(invalid_time,{d},.1);}catch(...){rejected=true;}require(rejected,"invalid time accepted");}
        bool rejected=false;try{irregular.step(time+.001,{d},-.1);}catch(...){rejected=true;}require(rejected,"invalid horizon accepted");}
      if(i>5)require(a.size()==1&&a[0].id==1&&cv::norm(a[0].velocity-b[0].velocity)<1e-12,"rejected input changed motion state");
    }
    require(std::abs(a[0].velocity.x-30)<.1,"irregular timestamps used fixed frame interval");
    require(std::abs(a[0].future_center.x-(100+30*(time+.1)))<.1,"irregular-time forecast wrong");
    const auto coast=irregular.step(time+.05,{},.1);
    require(coast.size()==1&&coast[0].id==1&&!coast[0].measurement_used,"irregular short gap lost track");
    const auto recovered=irregular.step(time+.09,{detection(float(100+30*(time+.09)))},.1);
    require(recovered.size()==1&&recovered[0].id==1&&recovered[0].measurement_used&&!recovered[0].future_valid,"irregular reacquisition did not reconfirm");
    auto fresh=irregular.step(time+1.,{detection(200)},.1);require(fresh.empty(),"expired identity bypassed confirmation");
    for(int i=1;i<=2;++i)fresh=irregular.step(time+1.+i*.03,{detection(200)},.1);
    require(fresh.size()==1&&fresh[0].id==2&&cv::norm(fresh[0].velocity)<1e-12,"expired track reused identity/velocity");
  }
  {TrackerConfig c;c.one_sided_guard_enabled=true;Tracker guarded(c),control(c);
    for(int i=0;i<10;++i){guarded.step(i/30.,{detection(100+i)},.1);control.step(i/30.,{detection(100+i)},.1);}
    for(int i=10;i<13;++i){auto partial=detection(100+i);
      partial.measurement_points[0].y+=5;partial.corners=partial.measurement_points;
      partial.center={0,0};for(auto p:partial.measurement_points)partial.center+=p*.25f;
      auto a=guarded.step(i/30.,{partial},.1),b=control.step(i/30.,{},.1);
      require(a.size()==1&&a[0].id==1&&!a[0].measurement_used&&!a[0].future_valid,"partial observation updated or spawned a track");
      require(cv::norm(a[0].filtered_center-b[0].filtered_center)<1e-9&&cv::norm(a[0].velocity-b[0].velocity)<1e-9,"rejected partial changed motion state");
      bool reason=false;for(auto d:guarded.association_diagnostics())reason|=std::string(d.reason)=="one_sided_length_drop";
      require(reason,"missing partial rejection provenance");
    }
    auto recovered=guarded.step(13/30.,{detection(113)},.1);
    require(recovered.size()==1&&recovered[0].id==1&&recovered[0].measurement_used&&!recovered[0].future_valid,"guard reacquisition bypassed confirmation");
    for(int i=14;i<18;++i)recovered=guarded.step(i/30.,{detection(100+i)},.1);
    require(recovered[0].future_valid,"guard recovery never reconfirmed prediction");
    require(guarded.step(1.0,{},.1).empty(),"guard prolonged expired track");
    Tracker scaled(c);for(int i=0;i<10;++i)scaled.step(i/30.,{detection(100)},.1);
    auto smaller=detection(100);for(auto& p:smaller.measurement_points)p.y=50+(p.y-50)*.65;
    smaller.corners=smaller.measurement_points;
    require(scaled.step(10/30.,{smaller},.1).at(0).measurement_used,"bilateral scale change treated as single-side clipping");
  }
  {TrackerConfig c;c.confirmation_hits=1;Tracker t(c);
    for(int i=0;i<12;++i){auto d=detection(100+i);
      if(i%2)for(auto& p:d.corners)p=d.center+(p-d.center)*4;
      const auto out=t.step(i/30.,{d},.1);
      require(out.size()==1&&out[0].id==1&&out[0].measurement_used,
              "display enclosure area changed measurement association");
    }
  }
  TrackerConfig config;Tracker tracker(config);std::vector<TrackOutput> out;
  auto quality_config=config;quality_config.confirmation_hits=1;quality_config.prediction_hits=1;
  quality_config.low_resolution_noise_scale=9;
  Tracker ordinary(quality_config),uncertain(quality_config),competition_quality(quality_config);
  for(int i=0;i<20;++i){ordinary.step(i/30.,{detection(100)},.1);
    uncertain.step(i/30.,{detection(100)},.1);competition_quality.step(i/30.,{detection(100)},.1);}
  auto low=detection(102);low.low_resolution=true;
  const auto normal_update=ordinary.step(20/30.,{detection(102)},.1).at(0);
  const auto weak_update=uncertain.step(20/30.,{low},.1).at(0);
  require(weak_update.measurement_low_resolution&&weak_update.measurement_variance==36,
          "low-resolution provenance/variance lost");
  require(weak_update.filtered_center.x<normal_update.filtered_center.x&&
          weak_update.position_std>normal_update.position_std,"larger R did not weaken update/increase uncertainty");
  low=detection(101);low.low_resolution=true;
  const auto competed=competition_quality.step(20/30.,{low,detection(101)},.1);
  bool kept=false;for(const auto& t:competed)if(t.id==1){kept=t.measurement_used&&!t.measurement_low_resolution;}
  require(kept,"larger R won association only by shrinking NIS");
  const auto coast_quality=uncertain.step(21/30.,{},.1).at(0);
  require(!coast_quality.measurement_low_resolution&&coast_quality.measurement_variance==0,
          "coast claims a current measurement");
  Tracker default_a(config),default_b(config);
  for(int i=0;i<20;++i){auto d=detection(100+i);auto marked=d;marked.low_resolution=true;
    const auto a=default_a.step(i/30.,{d},.1),b=default_b.step(i/30.,{marked},.1);
    require(a.size()==b.size(),"unit quality scale changed confirmation");
    if(!a.empty())require(cv::norm(a[0].future_center-b[0].future_center)==0&&a[0].innovation==b[0].innovation,
                          "unit quality scale changed baseline geometry");}
  auto diagnostic_config=config;diagnostic_config.confirmation_hits=1;diagnostic_config.prediction_hits=1;
  Tracker diagnostic(diagnostic_config);
  const auto initial=diagnostic.step(0,{detection(100)},.1).at(0);
  const double expected_variance=config.measurement_noise+.01*config.initial_velocity_variance+
                                 config.acceleration_noise*.001/3;
  require(std::abs(initial.future_position_variance[0]-expected_variance)<1e-10,
          "forecast diagnostic omits velocity/process uncertainty");
  for(int i=0;i<30;++i){
    const float x=100+i;
    out=tracker.step(i/30.,{detection(x)},.1);
    if(i<2)require(out.empty(),"tentative track rendered");
  }
  require(out.size()==1&&out[0].id==1&&out[0].measurement_used,"ID changed under constant motion");
  require(out[0].future_valid,"future missing after sufficient history");
  require(out[0].consecutive_hits==30,"consecutive diagnostic count wrong");
  require(std::abs(out[0].velocity.x-30)<1&&std::abs(out[0].velocity.y)<1e-6,"velocity diagnostic wrong");
  require(cv::norm(cv::Point2d(out[0].future_center)-(out[0].filtered_center+.1*out[0].velocity))<1e-4,
          "diagnostic state does not reproduce forecast");
  for(int axis=0;axis<2;++axis)require(std::isfinite(out[0].velocity_variance[axis])&&
      out[0].velocity_variance[axis]>0&&out[0].future_position_variance[axis]>0,"invalid diagnostic covariance");
  const double predicted=(out[0].future[0].x+out[0].future[1].x)/2;
  require(std::abs(predicted-132)<.6,"constant velocity prediction inaccurate");
  out=tracker.step(1.,{},.1);
  require(out.size()==1&&!out[0].measurement_used&&out[0].future_valid,"short occlusion not coasted");
  require(out[0].consecutive_hits==0,"coast did not reset consecutive diagnostic");
  require(out[0].current[0].x>119,"coasting position stopped");
  out=tracker.step(31/30.,{detection(131)},.1);
  require(out.size()==1&&out[0].id==1&&out[0].measurement_used,"short reacquisition lost ID");
  require(!out[0].future_valid,"reacquisition immediately reused unconfirmed velocity");
  require(out[0].consecutive_hits==1,"reacquisition diagnostic count wrong");
  out=tracker.step(32/30.,{detection(400)},.1);
  require(out.size()==1&&!out[0].measurement_used,"outlier used by old track");
  out=tracker.step(1.5,{},.1);require(out.empty(),"expired boxes still drawn");
  bool rejected=false;try{tracker.step(1.5,{},.1);}catch(const std::exception&){rejected=true;}
  require(rejected,"duplicate timestamp accepted");
  Tracker stationary(config);
  for(int i=0;i<30;++i)out=stationary.step(i/30.,{detection(100)},.1);
  require(cv::norm(out[0].future[0]-out[0].current[0])<.001,"artificial lead at rest");
  Tracker first(config),second(config);
  Tracker perspective(config);
  for(int i=0;i<70;++i){
    auto d=detection(100+i);
    const float halfwidth=2+18*std::abs(std::cos(i*.06));
    d.corners[0].x=d.corners[3].x=d.center.x-halfwidth;
    d.corners[1].x=d.corners[2].x=d.center.x+halfwidth;
    auto result=perspective.step(i/30.,{d},.1);
    if(i>=2)require(result.size()==1&&result[0].id==1&&result[0].measurement_used,
                   "perspective narrowing broke center association");
    if(!result.empty()&&result[0].future_valid){
      std::vector<cv::Point2f> p(result[0].future.begin(),result[0].future.end());
      require(cv::isContourConvex(p),"future quad inverted");
    }
  }
  Tracker expiry(config);
  for(int i=0;i<20;++i)expiry.step(i/30.,{detection(100+i)},.1);
  auto aged=expiry.step(19/30.+.25,{},.1);
  require(aged.empty()||(aged.size()==1&&!aged[0].future_valid),
          "future extends beyond coasting validity deadline");
  Tracker curved(config);
  double max_curve_error=0;
  for(int i=0;i<180;++i){
    const double time=i/30.;
    auto result=curved.step(time,{detection(float(150+60*std::sin(time*2)),
                                                float(100+20*std::cos(time*2)))},.1);
    if(i>=2)require(result.size()==1&&result[0].id==1&&result[0].measurement_used,
                     "curved motion fragmented into new IDs");
    if(i>10&&result[0].future_valid){
      const cv::Point2f truth(float(150+60*std::sin((time+.1)*2)),float(100+20*std::cos((time+.1)*2)));
      max_curve_error=std::max(max_curve_error,cv::norm(result[0].future_center-truth));
    }
  }
  require(max_curve_error<5,"curved future center error exceeds 5px on synthetic truth");
  Tracker ambiguous(config);
  for(int i=0;i<10;++i)ambiguous.step(i/30.,{detection(100),detection(108)},.1);
  auto ambiguous_result=ambiguous.step(10/30.,{detection(104)},.1);
  require(ambiguous_result.size()==2,"ambiguous input spawned duplicate identity");
  for(const auto& t:ambiguous_result)require(!t.measurement_used,"ambiguous observation forced into an ID");
  Tracker independent_center(config);
  for(int i=0;i<15;++i){auto d=detection(100);
    // Display padding changes; actual lamp measurement does not move.
    for(auto& p:d.corners)p.x+=i*.5f;
    auto r=independent_center.step(i/30.,{d},.1);
    if(i>=4)require(r.size()==1&&cv::norm(r[0].future_center-cv::Point2f(100,50))<.01,
                    "display contour contaminated center velocity");
  }
  for(int i=0;i<20;++i){auto a=first.step(i/30.,{detection(100+i)},.1);
    auto b=second.step(i/30.,{detection(100+i)},.5);
    require(a.size()==b.size(),"prediction mutated live state");
    if(!a.empty()){
      require(cv::norm(a[0].velocity-b[0].velocity)<1e-12,"diagnostic velocity depends on horizon");
      require(b[0].future_position_variance[0]>a[0].future_position_variance[0],"longer horizon uncertainty not growing");
    }
    if(!a.empty())require(cv::norm(a[0].current[0]-b[0].current[0])<1e-5,"horizon affected tracking");
  }
  std::cout<<"tracking, gating, occlusion, expiry, stationary and immutable prediction passed\n";
  Tracker crossing(config);
  auto competition_config=config;competition_config.confirmation_hits=1;
  competition_config.prediction_hits=1;competition_config.association_pixels=10;
  competition_config.ambiguity_margin=.1;
  Tracker competition(competition_config);
  competition.step(0,{detection(100),detection(110)},.1);
  const auto matched=competition.step(1/30.,{detection(102),detection(96)},.1);
  require(matched.size()==2,"candidate competition fragmented tracks");
  for(const auto& t:matched)require(t.measurement_used&&
      std::abs(t.current_center.x-(t.id==1?96:102))<.001,"greedy observation stealing remains in tracker");
  for(int i=0;i<41;++i){
    auto a=detection(80+2*i),b=detection(160-2*i);
    // Swap input ordering every frame. Physical motion, not vector order,
    // determines the identity after the ambiguous crossing.
    auto r=crossing.step(i/30.,i%2?std::vector<Detection>{b,a}:std::vector<Detection>{a,b},.1);
    if(i<2)continue;
    require(r.size()==2,"crossing created extra identities or lost a track");
    if(i==20)for(const auto& t:r)require(!t.measurement_used,"coincident crossing forced an identity");
    if(i>=25)for(const auto& t:r){
      require(t.measurement_used,"track did not reacquire after crossing");
      const float expected=t.id==1?80+2*i:160-2*i;
      require(std::abs(t.current_center.x-expected)<.001,"identity swapped after crossing");
    }
  }
}
