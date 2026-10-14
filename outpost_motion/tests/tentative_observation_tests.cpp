#include "target_model.hpp"
#include <stdexcept>
#include <iostream>
using namespace outpost;
void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
Detection pair(float x){Detection d;d.center={x,50};
 d.corners={cv::Point2f(x-10,40),{x+10,40},{x+10,60},{x-10,60}};d.measurement_points=d.corners;return d;}
int main(){try{
 TrackerConfig c;Tracker t(c);
 auto r=t.step(0,{pair(100)},.1);
 check(r.empty()&&t.tentative_observations().size()==1,"first accepted pair hidden");
 const auto id=t.tentative_observations()[0].id;
 check(t.tentative_observations()[0].detection_index==0,"wrong measurement index");
 r=t.step(.033,{pair(101)},.1);
 check(r.empty()&&t.tentative_observations().at(0).id==id,"tentative ID changed");
 r=t.step(.066,{pair(102)},.1);
 check(r.size()==1&&r[0].id==id&&!r[0].future_valid&&t.tentative_observations().empty(),"promotion duplicated observation or enabled forecast early");
 Tracker missing(c);missing.step(0,{pair(100)},.1);missing.step(.033,{},.1);
 check(missing.tentative_observations().empty(),"tentative missing frame drew a prediction");
 auto bad=pair(100);bad.measurement_points.fill({0,0});Tracker invalid(c);invalid.step(0,{bad},.1);
 check(invalid.tentative_observations().empty(),"invalid pair drawn");
 Tracker ambiguous(c);ambiguous.step(0,{pair(100),pair(108)},.1);ambiguous.step(.033,{pair(104)},.1);
 check(ambiguous.tentative_observations().empty(),"ambiguous rejected pair drawn");
 Tracker reset(c);reset.step(0,{pair(100)},.1);reset.reconfigure(c);
 check(reset.tentative_observations().empty(),"stale observation survived reconfigure");
 ImageTargetModel model(c,SlotConfig{});auto estimate=model.update(0,{pair(100)},.1);
 check(estimate.tracks.empty()&&estimate.slots.empty()&&estimate.tentative_observations.size()==1,
  "tentative observation entered confirmed slot resolver");
 std::cout<<"tentative real observation contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
