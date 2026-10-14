#include "prediction_audit.hpp"
#include <iostream>
#include <vector>
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){
  using namespace outpost;
  TrackOutput track;track.id=7;track.measurement_used=true;
  std::map<std::uint64_t,const TrackOutput*> observed{{7,&track}};
  std::vector<std::string> statuses;
  auto record=[&](const AuditedForecast&,const std::string& s,const TrackOutput* value){
    require((s=="scored")==bool(value),"unscored prediction got pseudo-ground-truth");statuses.push_back(s);
  };
  PredictionAudit audit;
  audit.observe(0,observed,record);audit.issue({0,2,7,{},{},{},true});
  audit.observe(1,observed,record);require(statuses.empty(),"forecast scored before deadline");
  audit.observe(2,observed,record);require(statuses.back()=="scored","continuous forecast not scored");
  audit.issue({2,4,7,{},{},{},true});audit.observe(3,{},record);audit.observe(4,observed,record);
  require(statuses.back()=="interrupted_history","reacquisition silently counted as continuous identity");
  audit.issue({4,5,7,{},{},{},true});audit.observe(5,{},record);
  require(statuses.back()=="no_accepted_target","missing measurement counted as correct");
  audit.issue({5,6,7,{},{},{},true});audit.observe(7,observed,record);
  require(statuses.back()=="timestamp_skipped","nearest frame substituted at wrong horizon");
  audit.issue({7,10,7,{},{},{},true});audit.finish(record);audit.finish(record);
  require(statuses.size()==5&&statuses.back()=="end_of_video","EOF forecast omitted or counted twice");
  bool rejected=false;try{audit.observe(7,observed,record);}catch(const std::exception&){rejected=true;}
  require(rejected,"duplicate frame accepted");
  std::cout<<"forecast deadline, continuity and complete accounting passed\n";
}
