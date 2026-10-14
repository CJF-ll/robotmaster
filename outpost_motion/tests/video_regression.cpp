#include "core.hpp"
#include <iostream>
#include <stdexcept>
#include <map>
#include <set>
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(int argc,char**argv){
  if(argc!=2)return 2;
  auto c=outpost::Config::load(argv[1]);outpost::VideoSource source(c);
  outpost::Detector detector(c.detector);outpost::Tracker tracker(c.tracker);
  outpost::Tracker padded_tracker(c.tracker);
  outpost::TargetSlots slots(c.slots);int slot_observations=0,consistent_observations=0;
  std::map<std::pair<int,std::uint64_t>,int> identities;
  std::set<int> learned_slots;
  int incomplete_sequence_checks=0;
  outpost::Frame frame;int n=0,measurements=0,observations=0,coasts=0,single=0,hidden=0;
  int failure_window_checks=0;bool saw_nonzero_prediction=false;
  std::optional<outpost::ImagePreprocessor> preprocessing;
  while(source.next(frame)){
    if(!preprocessing)preprocessing.emplace(c.geometry,frame.image.size());
    frame.image=preprocessing->process(frame.image);
    auto detections=detector.detect(frame.image);
    auto tracks=tracker.step(frame.time_seconds,detections,c.prediction_seconds);
    auto padded=detections;
    for(auto& d:padded)for(auto& p:d.corners)p=d.center+(p-d.center)*(frame.index%2?4.f:1.f);
    const auto padded_tracks=padded_tracker.step(frame.time_seconds,padded,c.prediction_seconds);
    require(tracks.size()==padded_tracks.size(),"display margin changed full-video track count");
    for(size_t i=0;i<tracks.size();++i){const auto& a=tracks[i];const auto& b=padded_tracks[i];
      require(a.id==b.id&&a.measurement_used==b.measurement_used&&a.detection_index==b.detection_index&&
              a.future_valid==b.future_valid&&cv::norm(a.future_center-b.future_center)<1e-5,
              "display margin changed full-video association or motion");
    }
    auto assignments=slots.update(frame.time_seconds,tracks,detections);
    int accepted_count=0,accepted_slots=0;std::set<int> frame_slots;
    for(size_t i=0;i<tracks.size();++i){
      accepted_count+=tracks[i].measurement_used;
      const int slot=assignments[i].slot;
      if(slot<0)continue;
      require(slot<c.slots.count&&frame_slots.insert(slot).second,"duplicate or invalid board identity");
      const auto key=std::make_pair(slots.epoch(),tracks[i].id);
      auto found=identities.find(key);
      require(found==identities.end()||found->second==slot,"local identity changed board within epoch");
      identities[key]=slot;learned_slots.insert(slot);
      accepted_slots+=tracks[i].measurement_used;
    }
    if(c.slots.enabled&&(accepted_count!=1||accepted_slots!=1)){
      ++incomplete_sequence_checks;
      require(!slots.consistent()&&slots.recurrence_seconds()==0,
              "incomplete observations asserted a current order or recurrence");
      for(const auto& a:assignments)require(!a.sequence_consistent,"stale per-slot sequence flag");
    }
    auto decisions=outpost::visibility_batch(tracks,detector.bars(),detector.pair_candidates(),c.visibility);
    const auto padded_decisions=outpost::visibility_batch(padded_tracks,detector.bars(),detector.pair_candidates(),c.visibility);
    for(size_t i=0;i<decisions.size();++i)
      require(decisions[i].current==padded_decisions[i].current&&
              decisions[i].future==padded_decisions[i].future&&
              decisions[i].evidence==padded_decisions[i].evidence,
              "display padding changed full-video visibility evidence");
    ++n;measurements+=int(detections.size());
    for(size_t i=0;i<tracks.size();++i){auto& t=tracks[i];auto& decision=decisions[i];
      if(t.measurement_used){++observations;
        if(assignments[i].slot>=0)++slot_observations;
        if(assignments[i].sequence_consistent)++consistent_observations;
        require(t.detection_index>=0&&size_t(t.detection_index)<detections.size(),"missing measurement provenance");
        const auto& d=detections[t.detection_index];
        require(cv::norm(t.current_center-d.center)<1e-4,"measurement center contaminated by display padding");
        for(int k=0;k<4;++k)require(cv::norm(t.current[k]-d.corners[k])<1e-4,"observed quad altered");
        for(int k=0;k<4;++k)require(cv::norm(t.lamp_support_points[k]-d.measurement_points[k])<1e-4,
                                  "lamp support geometry lost measurement provenance");
      }else ++coasts;
      if(decision.evidence==outpost::Evidence::None){++hidden;
        require(!decision.current&&!decision.future,"unsupported model box drawn");}
      if(decision.evidence==outpost::Evidence::CompatibleSingleLamp){++single;
        require(!t.measurement_used&&!decision.future,"single lamp presented as full measurement/future");}
      if(decision.future)require(t.measurement_used&&t.future_valid,"future lacks confirmed measurement history");
      if(frame.index>=192&&frame.index<=198&&!t.measurement_used&&
         t.current_center.y>185&&t.current_center.y<215&&t.current_center.x<310){
        ++failure_window_checks;require(!decision.current&&!decision.future,"known drifting box returned");
      }
      if(t.future_valid&&cv::norm(t.future_center-t.current_center)>5)saw_nonzero_prediction=true;
    }
  }
  require(n==925,"wrong regression video");require(measurements>800,"detector coverage collapsed");
  require(observations>700&&failure_window_checks>0&&saw_nonzero_prediction,"regression did not exercise required paths");
  if(c.slots.enabled)require(slot_observations>600&&int(learned_slots.size())==c.slots.count&&incomplete_sequence_checks>0,
                             "real-video identity coverage or missing-evidence checks collapsed");
  std::cout<<"frames="<<n<<" candidates="<<measurements<<" observations="<<observations
           <<" coasts="<<coasts<<" single="<<single<<" hidden="<<hidden
           <<" known_failure_checks="<<failure_window_checks
           <<" slot_observations="<<slot_observations<<" sequence_consistent="<<consistent_observations
           <<" incomplete_sequence_checks="<<incomplete_sequence_checks<<'\n';
}
