#pragma once
#include "tracker.hpp"
#include "target_slots.hpp"
#include <memory>
#include <string>
#include <stdexcept>
namespace outpost {
struct TargetEstimate {
  CoaxialState coaxial;
  std::vector<TrackOutput> tracks;
  std::vector<Tracker::TentativeObservation> tentative_observations;
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
  ImageTargetModel(TrackerConfig t,SlotConfig s):tracker_(t),slots_(s),config_(t),coaxial_(t.coaxial){}
  TargetEstimate update(double time,const std::vector<Detection>& observations,double horizon)override{
    TargetEstimate e;e.tracks=tracker_.step(time,observations,horizon);
    e.tentative_observations=tracker_.tentative_observations();
    e.slots=slots_.update(time,e.tracks,observations);e.associations=tracker_.association_diagnostics();
    e.epoch=slots_.epoch();e.sequence_consistent=slots_.consistent();e.recurrence_seconds=slots_.recurrence_seconds();
    if(config_.coaxial.enabled){
      if(coaxial_epoch_!=e.epoch){coaxial_=CoaxialPhaseModel(config_.coaxial);coaxial_epoch_=e.epoch;}
      std::vector<CoaxialObservation> accepted;
      std::vector<int> phase_input(e.tracks.size(),-1);
      for(size_t i=0;i<e.tracks.size();++i)if(e.tracks[i].measurement_used){
        phase_input[i]=int(accepted.size());
        accepted.push_back({e.slots[i].slot,e.tracks[i].current_center});
      }
      // A mature common-axis model can uniquely associate an accepted real
      // pair before its NEW local track has accumulated confirmation hits.
      // It does not promote that local ID or create a tentative future box.
      for(const auto& o:e.tentative_observations)
        accepted.push_back({-1,observations.at(size_t(o.detection_index)).center});
      e.coaxial=coaxial_.update(time,accepted,horizon);
      for(size_t i=0;i<e.tracks.size();++i){auto& track=e.tracks[i];const int slot=e.slots[i].slot;
        // No silent linear fallback while the selected coaxial model warms up.
        track.future_center_valid=false;track.future_valid=false;track.center_forecast_blended=false;
        if(slot<0||!e.coaxial.valid||!e.coaxial.measurement_used||!track.measurement_used||
           phase_input[i]<0||e.coaxial.observation_slots.at(size_t(phase_input[i]))!=slot){
          track.forecast.blocks|=ForecastHistory;continue;
        }
        const auto& p=e.coaxial.future[size_t(slot)];
        track.future_center=cv::Point2f(float(p.x),float(p.y));
        const auto shift=track.future_center-track.current_center;
        for(int k=0;k<4;++k)track.future[k]=track.current[k]+shift;
        track.future_position_variance=e.coaxial.future_variance[size_t(slot)];
        track.forecast=assess_forecast(true,!(track.forecast.blocks&ForecastGeometry),track.future_position_variance,0,horizon,
          config_.coaxial.coast_seconds,config_.max_position_std,!(track.forecast.blocks&ForecastIdentity));
        track.future_center_valid=track.forecast.usable();track.future_valid=track.future_center_valid;
        if(config_.shape_guard_enabled&&!track.future_shape_reliable){track.forecast.blocks|=ForecastShape;track.future_valid=false;}
      }
    }
    return e;
  }
  void reconfigure(TrackerConfig t,SlotConfig s)override{tracker_.reconfigure(t);slots_.reconfigure(s);config_=t;coaxial_=CoaxialPhaseModel(t.coaxial);coaxial_epoch_=-1;}
 private:Tracker tracker_;TargetSlots slots_;TrackerConfig config_;CoaxialPhaseModel coaxial_;int coaxial_epoch_=-1;
};
inline std::unique_ptr<TargetModel> make_target_model(const std::string& mode,TrackerConfig t,SlotConfig s){
  if(mode=="image_2d")return std::make_unique<ImageTargetModel>(t,s);
  throw std::runtime_error("target estimator unavailable: "+mode+"; calibrated geometry alone is not an estimator");
}
}
