#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace outpost {
struct SemanticPolicy {
  int target=6,confirm=3,max_hold=1;
  double accept=.7,target_floor=.2,negative_veto=.9;
  void validate() const {
    if(target<0||target>=9||confirm<1||max_hold<0||
       !std::isfinite(accept)||!std::isfinite(target_floor)||!std::isfinite(negative_veto)||
       accept<=0||accept>1||target_floor<0||target_floor>accept||negative_veto<=0||negative_veto>1)
      throw std::runtime_error("invalid semantic policy");
  }
};
struct SemanticHistory {int confirmations=0,held=0;};
struct SemanticDecision {SemanticHistory next;bool direct=false,accepted=false;const char* reason="rejected";};
// Experimental only. History is usable only after externally verified unique
// consecutive-frame correspondence. This is NOT a physical identity tracker.
inline SemanticDecision semantic_step(const SemanticPolicy& policy,SemanticHistory previous,
    bool continuous,bool measured_support,const std::array<double,9>& p){
  policy.validate();SemanticDecision result;
  double sum=0;
  for(double v:p){if(!std::isfinite(v)||v<0||v>1){result.reason="invalid_scores";return result;}sum+=v;}
  if(std::abs(sum-1)>1e-5){result.reason="invalid_scores";return result;}
  const int best=int(std::max_element(p.begin(),p.end())-p.begin());
  result.direct=best==policy.target&&p[best]>=policy.accept;
  if(!measured_support){result.reason="no_measured_support";return result;}
  if(!continuous)previous={};
  if(result.direct){result.accepted=true;result.next={std::min(policy.confirm,previous.confirmations+1),0};result.reason="direct";return result;}
  double other=0;for(int i=0;i<9;++i)if(i!=policy.target)other=std::max(other,p[i]);
  if(other>=policy.negative_veto){result.reason="strong_contradiction";return result;}
  if(previous.confirmations>=policy.confirm&&previous.held<policy.max_hold&&p[policy.target]>=policy.target_floor){
    result.accepted=true;result.next={previous.confirmations,previous.held+1};result.reason="history_supported";
  }
  return result;
}
} // namespace outpost
