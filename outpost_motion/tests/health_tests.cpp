#include "innovation_health.hpp"
#include "tracker.hpp"
#include <stdexcept>
#include <limits>
using namespace outpost;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
Detection detection(float x){Detection d;d.center={x,50};
  d.measurement_points={cv::Point2f(x-10,44),{x+10,44},{x+10,56},{x-10,56}};
  d.corners=d.measurement_points;return d;}
int main(){
  HealthConfig c;c.min_samples=3;InnovationHealth h;
  check(!h.summary(c).ready,"empty evidence ready");
  h.add(0,{4,0},{4,4},c);h.add(.1,{4,0},{4,4},c);
  check(!h.summary(c).ready,"insufficient evidence ready");
  h.add(.2,{4,0},{4,4},c);
  auto s=h.summary(c);check(s.ready&&!s.warning&&s.mean_nis==4&&s.mean_standardized_residual.x==2,"wrong whitening");
  h.clear();for(int i=0;i<3;++i)h.add(i*.1,{6,0},{4,4},c);
  check(h.summary(c).warning,"persistent large innovation not reported");
  h.add(1,{0,0},{4,4},c);check(h.summary(c).samples==1&&!h.summary(c).ready,"time window stale");
  bool rejected=false;try{h.add(1,{0,0},{4,4},c);}catch(...){rejected=true;}
  check(rejected,"non-increasing timestamp accepted");
  rejected=false;try{h.add(2,{0,0},{0,4},c);}catch(...){rejected=true;}
  check(rejected,"zero variance accepted");
  TrackerConfig a;a.confirmation_hits=1;auto b=a;b.health.min_samples=2;b.health.nis_warning=.000001;
  Tracker x(a),y(b);
  for(int i=0;i<20;++i){const auto p=x.step(i/30.,{detection(float(100+i*i*.1))},.1);
    const auto q=y.step(i/30.,{detection(float(100+i*i*.1))},.1);
    check(p.size()==q.size()&&p.size()==1,"health affected association");
    check(p[0].future_center==q[0].future_center&&p[0].future_valid==q[0].future_valid,"health affected forecast");
    if(i==0)check(p[0].health.samples==0,"birth counted as zero innovation");
  }
  const auto coast=x.step(20/30.,{},.1);
  check(coast.size()==1&&coast[0].health.samples==0,"miss retained consecutive health");
  check(x.association_diagnostics().size()==1&&std::string(x.association_diagnostics()[0].reason)=="no_candidates","missing no-candidate reason");
  Tracker gates(a);gates.step(0,{detection(100)},.1);
  gates.step(.03,{detection(500)},.1);
  check(std::string(gates.association_diagnostics()[0].reason)=="distance_gate"&&
        !gates.association_diagnostics()[0].innovation_available,"wrong gate provenance");
  auto invalid=detection(500);invalid.measurement_points[0]=invalid.measurement_points[1];
  Tracker bad(a);check(bad.step(0,{invalid},.1).empty(),"degenerate measurement spawned track");
}
