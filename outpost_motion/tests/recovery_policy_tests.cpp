#include "tracker.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace outpost;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Detection observation(float x=100){
  Detection d;d.center={x,50};d.corners={cv::Point2f(x-10,44),{x+10,44},{x+10,56},{x-10,56}};
  d.measurement_points=d.corners;return d;
}
TrackOutput one(const std::vector<TrackOutput>& tracks){
  check(tracks.size()==1,"expected one retained track");return tracks[0];
}
const TrackOutput& id_one(const std::vector<TrackOutput>& tracks){
  for(const auto& t:tracks)if(t.id==1)return t;
  throw std::runtime_error("original identity disappeared");
}
void warm(Tracker& tracker){for(int i=0;i<10;++i)tracker.step(i/30.,{observation()},.1);}
void same_state(const TrackOutput& a,const TrackOutput& b){
  check(a.id==b.id&&a.measurement_used==b.measurement_used&&a.current==b.current&&a.future==b.future&&
        a.filtered_center==b.filtered_center&&a.velocity==b.velocity&&
        a.future_position_variance==b.future_position_variance&&a.velocity_variance==b.velocity_variance,
        "recovery admission changed association, motion or geometry");
}
int main(){try{
  TrackerConfig config;config.clean_recovery_enabled=true;
  auto legacy=config;legacy.clean_recovery_enabled=false;
  for(int missing:{1,2,3}){
    Tracker fast(config),baseline(legacy);warm(fast);warm(baseline);
    for(int i=10;i<10+missing;++i){auto a=one(fast.step(i/30.,{},.1)),b=one(baseline.step(i/30.,{},.1));
      same_state(a,b);check(!a.measurement_used&&a.consecutive_hits==0&&!a.clean_recovery_used,
                           "missing observations counted as recovery evidence");}
    for(int k=1;k<=5;++k){const double time=(9+missing+k)/30.;
      const auto a=one(fast.step(time,{observation()},.1)),b=one(baseline.step(time,{observation()},.1));
      same_state(a,b);check(a.consecutive_hits==k,"fast path changed genuine consecutive counter");
      const bool eligible=missing<=2;
      check(a.future_valid==(k>=(eligible?2:5)),"short-gap recovery/deadline incorrect");
      check(b.future_valid==(k>=5),"disabled policy changed ordinary recovery");
      check(a.clean_recovery_used==(eligible&&k==2),"recovery event missing or repeated");
    }
  }
  // Time is checked before last_observed is overwritten, including the exact boundary.
  for(double gap:{.1,.10001}){
    Tracker t(config);warm(t);t.step(.33,{},.1);
    t.step(.3+gap,{observation()},.1);const auto out=one(t.step(.3+gap+1/30.,{observation()},.1));
    check(out.future_valid==(gap==.1),"gap check used refreshed timestamp");
  }
  {Tracker t(config);for(int i=0;i<3;++i)t.step(i/30.,{observation()},.1);
    t.step(.1,{},.1);t.step(4/30.,{observation()},.1);
    check(!one(t.step(5/30.,{observation()},.1)).future_valid,"immature track borrowed recovery history");}
  {Tracker t(config);warm(t);t.step(10/30.,{},.1);t.step(11/30.,{observation()},.1);t.step(12/30.,{},.1);
    for(int k=1;k<=5;++k){const auto out=one(t.step((12+k)/30.,{observation()},.1));
      check(out.future_valid==(k==5)&&!out.clean_recovery_used,"interrupted recovery kept fast eligibility");}}
  // Strong but still accepted residuals cannot qualify just because the next one fits.
  for(int bad_hit:{1,2}){auto c=config;c.clean_recovery_max_nis=1e-8;Tracker t(c);warm(t);t.step(10/30.,{},.1);
    for(int k=1;k<=5;++k){const auto out=one(t.step((10+k)/30.,{observation(k>=bad_hit?102:100)},.1));
      check(out.measurement_used,"test residual failed the ordinary association gate");
      check(out.future_valid==(k==5)&&!out.clean_recovery_used,"large residual retained fast eligibility");}}
  {Tracker t(config);warm(t);t.step(10/30.,{},.1);auto low=observation();low.low_resolution=true;
    t.step(11/30.,{low},.1);
    check(!one(t.step(12/30.,{observation()},.1)).future_valid,"low-resolution R diluted recovery gate");}
  // A nonempty rejected candidate makes the outage dirty even if a later frame is empty.
  for(int kind=0;kind<3;++kind){Tracker t(config);warm(t);auto bad=observation();
    if(kind==0)bad=observation(500);
    if(kind==1)bad.center.x=std::numeric_limits<float>::quiet_NaN();
    if(kind==2){bad.measurement_points[0].y=bad.measurement_points[3].y;bad.corners=bad.measurement_points;}
    t.step(10/30.,{bad},.1);t.step(11/30.,{},.1);
    t.step(12/30.,{observation()},.1);const auto out=t.step(13/30.,{observation()},.1);
    check(!id_one(out).future_valid&&!id_one(out).clean_recovery_used,"dirty rejection became clean gap");}
  {auto c=config;c.one_sided_guard_enabled=true;Tracker t(c);warm(t);auto partial=observation();
    partial.measurement_points[0].y+=5;partial.corners=partial.measurement_points;
    partial.center={0,0};for(auto p:partial.measurement_points)partial.center+=p*.25f;
    t.step(10/30.,{partial},.1);t.step(11/30.,{observation()},.1);
    check(!one(t.step(12/30.,{observation()},.1)).future_valid,"quality rejection used fast recovery");}
  // Multiple eligible edges, even a unique optimal solution, prohibit fast admission.
  for(bool disputed:{false,true}){Tracker t(config);
    for(int i=0;i<10;++i)t.step(i/30.,{observation(100),observation(108)},.1);
    t.step(10/30.,disputed?std::vector<Detection>{observation(104)}:std::vector<Detection>{},.1);
    for(int k=1;k<=5;++k){const auto out=t.step((10+k)/30.,{observation(100),observation(108)},.1);
      check(out.size()==2,"competition changed number of identities");
      for(const auto& o:out){check(o.measurement_used&&o.current_center.x==(o.id==1?100:108),"competition misassociated identity");
        check(o.future_valid==(k==5)&&!o.clean_recovery_used,"competition or ambiguity used fast recovery");}}
  }
  {Tracker t(config);warm(t);t.step(10/30.,{},.1);check(t.step(1.,{},.1).empty(),"fast recovery extended expiry");
    for(int i=0;i<5;++i){const auto out=t.step(1.1+i/30.,{observation()},.1);
      if(i>=2)check(out.size()==1&&out[0].id==2&&out[0].future_valid==(i==4)&&!out[0].clean_recovery_used,
                     "new identity inherited mature recovery status");}}
  {Tracker short_horizon(config),long_horizon(config);warm(short_horizon);warm(long_horizon);
    short_horizon.step(10/30.,{},.1);long_horizon.step(10/30.,{},.4);
    for(int k=1;k<=2;++k){const auto a=one(short_horizon.step((10+k)/30.,{observation()},.1));
      const auto b=one(long_horizon.step((10+k)/30.,{observation()},.4));
      check(a.velocity==b.velocity&&a.filtered_center==b.filtered_center,"horizon changed recovery state");
      check(!b.future_valid&&(b.forecast.blocks&ForecastDeadline),"fast recovery bypassed deadline");}}
  std::cout<<"clean recovery: 1/2-frame gaps need 2 true hits; 3-frame, dirty, competing and interrupted gaps require full history\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
