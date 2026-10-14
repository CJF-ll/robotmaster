#include "coaxial_phase.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace outpost;
void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
cv::Point2d point(int slot,double phase){const double a=phase+slot*2*CV_PI/3;
  return {300+80*std::cos(a)+7*slot,150+18*slot+15*std::sin(a)};}
int main(){
  for(double velocity:{-.9,1.7,-3.2}){
    CoaxialConfig c;CoaxialPhaseModel model(c);int valid=0;
    for(int i=0;i<180;++i){const double t=i/30.;const int slot=i%3;
      auto p=model.update(t,{{slot,point(slot,.3+velocity*t)}},.1);
      if(p.valid){++valid;require(std::abs(p.omega-velocity)<.08,"wrong phase speed/alias");
        for(int s=0;s<3;++s){
          require(cv::norm(p.model_point(s,p.phase)-p.current[s])<1e-8,"model/current geometry mismatch");
          require(cv::norm(p.model_point(s,p.phase+p.omega*.1)-p.future[s])<1e-8,"model/future geometry mismatch");
          require(cv::norm(p.model_point(s,p.phase+2*CV_PI)-p.current[s])<1e-8,"model orbit not closed");
        }
        for(int s=0;s<3;++s)require(cv::norm(p.future[s]-point(s,.3+velocity*(t+.1)))<2,"coaxial phase projection error");
        require(std::abs(p.period_seconds-2*CV_PI/std::abs(velocity))<.6,"period confused with board interval");
      }
    }
    require(valid>70,"coaxial initialization did not converge");
  }
  CoaxialConfig c;CoaxialPhaseModel a(c),b(c);
  for(int i=0;i<160;++i){const double t=i/30.;const int s=i%3;
    auto x=a.update(t,{{s,point(s,.2+1.7*t)}},.1),y=b.update(t,{{s,point(s,.2+1.7*t)}},.1);
    require(x.valid==y.valid&&x.phase==y.phase&&x.omega==y.omega,"prefix causality failure");
  }
  auto coast=a.update(160/30.,{},.1);require(coast.initialized&&coast.valid&&!coast.measurement_used,"short missing frame not coasting");
  auto unknown=b.update(160/30.,{{-1,point(1,.2+1.7*160/30.)}},.1);
  require(unknown.measurement_used&&unknown.observation_slots==std::vector<int>{1},"unique unlabeled pair was not phase-associated");
  const auto p=point(2,.2+1.7*161/30.);
  auto ambiguous=b.update(161/30.,{{-1,p},{-1,p}},.1);
  require(!ambiguous.measurement_used,"ambiguous duplicates updated phase");
  for(int i=161;i<175;++i)coast=a.update(i/30.,{},.1);
  require(!coast.valid&&!coast.initialized,"expired phase persisted");
  CoaxialPhaseModel stopped(c);CoaxialState last;
  for(int i=0;i<210;++i){const double t=i/30.;const int s=i%3;
    last=stopped.update(t,{{s,point(s,.2+1.7*std::min(t,5.))}},.1);
  }
  require(last.valid&&std::abs(last.omega)<.07&&last.period_seconds==0,"stop retained fabricated finite period");
  CoaxialPhaseModel still(c);
  for(int i=0;i<150;++i)require(!still.update(i/30.,{{i%3,point(i%3,.2)}},.1).valid,"static bootstrap invented rotation");
  bool rejected=false;try{still.update(5,{{0,{1,2}},{0,{3,4}}},.1);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"duplicate slot accepted");
  c.max_omega=c.min_omega;require(!coaxial_config_valid(c),"invalid frequency range accepted");
  std::cout<<"coaxial online phase contracts passed\n";
}
