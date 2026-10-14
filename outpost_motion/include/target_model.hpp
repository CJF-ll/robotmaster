#pragma once
#include "tracker.hpp"
#include "target_slots.hpp"
#include <memory>
#include <string>
#include <stdexcept>
namespace outpost {
struct TargetEstimate {
  std::vector<TrackOutput> tracks;
  std::vector<SlotAssignment> slots;
  std::vector<Tracker::AssociationDiagnostic> associations;
  int epoch=0;
  bool sequence_consistent=false;
  double recurrence_seconds=0;
  // Image-space tracks and slot hypotheses are NOT a calibrated rigid estimate.
  bool physical_state_available=false;
};
class TargetModel {
 public:
  virtual ~TargetModel()=default;
  virtual TargetEstimate update(double time,const std::vector<Detection>& observations,double horizon)=0;
  virtual void reconfigure(TrackerConfig tracker,SlotConfig slots)=0;
};
class ImageTargetModel final:public TargetModel {
 public:
  ImageTargetModel(TrackerConfig t,SlotConfig s):tracker_(t),slots_(s){}
  TargetEstimate update(double time,const std::vector<Detection>& observations,double horizon)override{
    TargetEstimate e;e.tracks=tracker_.step(time,observations,horizon);
    e.slots=slots_.update(time,e.tracks,observations);e.associations=tracker_.association_diagnostics();
    e.epoch=slots_.epoch();e.sequence_consistent=slots_.consistent();e.recurrence_seconds=slots_.recurrence_seconds();
    return e;
  }
  void reconfigure(TrackerConfig t,SlotConfig s)override{tracker_.reconfigure(t);slots_.reconfigure(s);}
 private:Tracker tracker_;TargetSlots slots_;
};
inline std::unique_ptr<TargetModel> make_target_model(const std::string& mode,TrackerConfig t,SlotConfig s){
  if(mode=="image_2d")return std::make_unique<ImageTargetModel>(t,s);
  throw std::runtime_error("target estimator unavailable: "+mode+"; calibrated geometry alone is not an estimator");
}
}
