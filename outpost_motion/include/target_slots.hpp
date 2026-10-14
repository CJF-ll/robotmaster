#pragma once
#include "tracker.hpp"
#include "slot_sequence.hpp"
#include <array>
#include <deque>
#include <map>
#include <vector>
namespace outpost {
struct SlotConfig {
  bool enabled=true;
  int count=3,local_observations=3, profile_samples=120, sequence_transitions=6;
  double height_gate_lengths=.65, length_ratio=1.8, angle_degrees=35;
  double ambiguity_margin=.2, reset_gap_seconds=5;
};
struct SlotAssignment {
  int slot=-1; // learned order of first encounters; never sorted by image y
  bool sequence_consistent=false;
};
// Image-space re-identification hypotheses for one fixed-camera target.
// These profiles are learned online. They are not metric heights or poses,
// and this module does not generate positions for invisible plates.
class TargetSlots {
 public:
  explicit TargetSlots(SlotConfig config):config_(config),sequence_(config.count,config.sequence_transitions){}
  void reconfigure(SlotConfig config);
  std::vector<SlotAssignment> update(double time,const std::vector<TrackOutput>&,
                                     const std::vector<Detection>&);
  int epoch() const { return epoch_; }
  bool consistent() const { return sequence_.consistent(); }
  double recurrence_seconds() const { return sequence_.recurrence_seconds(); }
 private:
  struct Feature { double x,y,length,angle; };
  struct Profile { std::deque<Feature> samples; };
  struct Local {
    int slot=-1,locked_slot=-1,observations=0,pending_slot=-1,pending_hits=0;
    double last_seen=-1;
    void clear_pending(){pending_slot=-1;pending_hits=0;}
  };
  SlotConfig config_;
  std::vector<Profile> profiles_;
  std::map<std::uint64_t,Local> locals_;
  SlotSequence sequence_;
  int epoch_=0;
  double last_time_=-1,last_observation_=-1;
  static Feature feature(const Detection&);
  double cost(const Profile&,const Feature&)const;
};
}
