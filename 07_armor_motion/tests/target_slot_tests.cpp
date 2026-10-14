#include "target_slots.hpp"
#include <stdexcept>
#include <iostream>
using namespace outpost;
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
Detection observation(float y,float x=100){
  Detection d;d.center={x,y};d.measurement_points={cv::Point2f(x-10,y-8),{x+10,y-8},{x+10,y+8},{x-10,y+8}};
  d.corners=d.measurement_points;return d;
}
int main(){
  SlotConfig cfg;cfg.local_observations=1;TargetSlots slots(cfg);double time=0;
  TrackOutput old_track;old_track.id=1;old_track.measurement_used=true;old_track.detection_index=0;
  auto new_track=old_track;new_track.id=2;
  TargetSlots rejected_birth(cfg);
  const auto first_conflict=rejected_birth.update(0,{old_track,new_track},{observation(100)});
  require(first_conflict[0].slot<0&&first_conflict[1].slot<0,"new-profile conflict setup failed");
  auto third=old_track;third.id=3;
  require(rejected_birth.update(.1,{third},{observation(160)})[0].slot==0,
          "rejected candidates permanently consumed a slot profile");
  TargetSlots mixed_birth(cfg);third.detection_index=1;
  const auto mixed=mixed_birth.update(0,{old_track,new_track,third},{observation(100),observation(160)});
  require(mixed[0].slot<0&&mixed[1].slot<0&&mixed[2].slot==0,
          "accepted new profile not remapped after rejected tentative profile");
  require(mixed_birth.update(.1,{third},{observation(100),observation(160)})[0].slot==0,
          "committed profile mapping not stable on next observation");
  TargetSlots collision(cfg);
  require(collision.update(0,{old_track},{observation(100)})[0].slot==0,"collision setup failed");
  require(collision.update(.1,{new_track},{observation(100)})[0].slot==0,"return setup failed");
  auto disputed=collision.update(.2,{old_track,new_track},{observation(100)});
  require(disputed[0].slot<0&&disputed[1].slot<0,"observed conflict not rejected");
  old_track.measurement_used=false;new_track.measurement_used=false;
  const auto after_dispute=collision.update(.3,{old_track,new_track},{});
  require(after_dispute[0].slot<0&&after_dispute[1].slot<0,"coasting resurrected disputed slot identity");
  old_track.measurement_used=true;
  require(collision.update(.4,{old_track},{observation(100)})[0].slot==0,"valid re-observation cannot recover locked identity");
  TargetSlots overlap(cfg);
  overlap.update(0,{old_track},{observation(100)});
  old_track.measurement_used=false;new_track.measurement_used=true;
  const auto overlapping=overlap.update(.1,{old_track,new_track},{observation(100)});
  require(overlapping[0].slot<0&&overlapping[1].slot==0,"coast duplicated freshly observed physical-slot hypothesis");
  new_track.measurement_used=false;
  require(overlap.update(.2,{old_track,new_track},{})[0].slot<0,"evicted coast slot returned without new evidence");
  TargetSlots invalid_measurement(cfg);old_track.measurement_used=true;old_track.detection_index=0;
  invalid_measurement.update(0,{old_track},{observation(100)});
  old_track.detection_index=99;
  require(invalid_measurement.update(.1,{old_track},{observation(100)})[0].slot<0,"invalid measurement index accepted");
  old_track.measurement_used=false;
  require(invalid_measurement.update(.2,{old_track},{})[0].slot<0,"invalid observation retained reusable slot");
  TargetSlots coast_conflict(cfg);old_track.measurement_used=true;old_track.detection_index=0;
  new_track.measurement_used=true;
  coast_conflict.update(0,{old_track},{observation(100)});
  coast_conflict.update(.1,{new_track},{observation(100)});
  old_track.measurement_used=false;new_track.measurement_used=false;
  auto coast_labels=coast_conflict.update(.2,{new_track,old_track},{});
  require(coast_labels[0].slot<0&&coast_labels[1].slot<0,"two coast labels claimed same slot");
  for(int event=0;event<12;++event){
    // Encounter order deliberately differs from vertical order.
    const float heights[]={100,160,130};const int expected=event%3;
    TrackOutput t;t.id=event+1;t.measurement_used=true;t.detection_index=0;
    auto result=slots.update(time,{t},{observation(heights[expected])});time+=.5;
    require(result[0].slot==expected,"returning board lost encounter-order slot");
    if(event<6)require(!result[0].sequence_consistent,"identity confirmed without repeated cycle");
  }
  require(slots.consistent(),"repeated ordered visits did not confirm cycle");
  require(std::abs(slots.recurrence_seconds()-1.5)<1e-6,"observed recurrence wrong");
  TrackOutput t;t.id=100;t.measurement_used=true;t.detection_index=0;
  auto unknown=slots.update(time,{t},{observation(210)});time+=.5;
  require(unknown[0].slot<0,"unmatched fourth geometry forced into a slot");
  t.id=101;auto conflicting=slots.update(time,{t,t},{observation(100)});time+=.5;
  require(conflicting[0].slot<0&&conflicting[1].slot<0,"simultaneous tracks claimed same slot");
  const int epoch=slots.epoch();slots.update(time+6,{},{});
  require(slots.epoch()==epoch+1&&!slots.consistent(),"long evidence gap retained confident identity");
  TargetSlots no_remap(cfg);t.id=1;
  require(no_remap.update(0,{t},{observation(100)})[0].slot==0,"initial slot missing");
  t.id=2;no_remap.update(.5,{t},{observation(130)});
  t.id=1;require(no_remap.update(1,{t},{observation(130)})[0].slot<0,"local track relabeled on geometry jump");
  require(no_remap.update(1.5,{t},{observation(130)})[0].slot<0,"repeated geometry jump relabeled track");
  TargetSlots slower(cfg);
  for(int event=0;event<12;++event){const float heights[]={100,160,130};
    t.id=event+1;slower.update(event*1.,{t},{observation(heights[event%3])});}
  require(std::abs(slower.recurrence_seconds()-3)<1e-6,"recurrence speed was preset");
  SlotConfig two_config=cfg;two_config.count=2;TargetSlots two(two_config);
  for(int event=0;event<10;++event){t.id=event+1;
    auto r=two.update(event*.5,{t},{observation(event%2?140:100)});
    require(r[0].slot==event%2,"configured slot count ignored");}
  require(two.consistent(),"two-slot repeating sequence not recognized");
  TargetSlots unmatched(cfg);t.id=1;unmatched.update(0,{t},{observation(100)});
  t.id=2;unmatched.update(.1,{t},{observation(130)});
  t.id=3;unmatched.update(.2,{t},{observation(160)});
  const int initial_epoch=unmatched.epoch();
  for(int i=1;i<=7;++i){t.id=10+i;unmatched.update(i,{t},{observation(250)});}
  require(unmatched.epoch()>initial_epoch,"unrelated detections kept stale identity alive");
  {
    SlotConfig recovery_cfg;recovery_cfg.local_observations=3;recovery_cfg.height_gate_lengths=1.1;
    TargetSlots recovery(recovery_cfg);double now=0;
    auto visit=[&](int id,float y,bool measured=true){
      TrackOutput track;track.id=id;track.measurement_used=measured;track.detection_index=0;
      const auto result=recovery.update(now,{track},measured?std::vector<Detection>{observation(y)}:std::vector<Detection>{});
      now+=.1;return result[0].slot;
    };
    for(int s=0;s<3;++s)for(int hit=0;hit<3;++hit)
      require(visit(s+1,100+30*s)==(hit==2?s:-1),"new profile bypassed confirmation");
    for(int hit=0;hit<6;++hit)
      require(visit(20,hit%2?130:100)<0,"alternating recovery geometry acquired an identity");
    require(visit(20,100)<0&&visit(20,100)<0&&visit(20,100)==0,
            "stable recovery did not confirm after three consistent measurements");
    require(visit(20,100,false)==0&&visit(20,100)==0,"short coast lost an established identity");
    require(visit(20,130)<0,"established identity switched to another board");
    require(visit(20,100)<0&&visit(20,100)<0&&visit(20,100)==0,
            "rejected geometry recovered identity without renewed evidence");
    require(visit(30,160)<0&&visit(30,160)<0,"premature recovery");
    require(visit(30,160,false)<0,"prediction counted as recovery evidence");
    require(visit(30,160)<0&&visit(30,160)<0&&visit(30,160)==2,
            "missed observation did not restart confirmation");
    require(visit(40,130)<0&&visit(40,130)<0,"premature absent-track recovery");
    recovery.update(now,{},{});now+=.1;
    require(visit(40,130)<0&&visit(40,130)<0&&visit(40,130)==1,
            "absent track retained consecutive recovery evidence");
    require(visit(50,100)<0&&visit(50,100)<0&&visit(50,115)<0,
            "ambiguous midpoint was assigned to a board");
    require(visit(50,100)<0&&visit(50,100)<0&&visit(50,100)==0,
            "ambiguity did not restart recovery confirmation");
    TrackOutput a;a.id=60;a.measurement_used=true;a.detection_index=0;
    auto b=a;b.id=61;
    for(int hit=0;hit<3;++hit){auto r=recovery.update(now,{a,b},{observation(130)});now+=.1;
      require(r[0].slot<0&&r[1].slot<0,"competing recoveries claimed the same board");}
    require(visit(60,130)<0&&visit(60,130)<0&&visit(60,130)==1,
            "conflict did not restart recovery confirmation");
  }
  std::cout<<"causal slot visits, cycle evidence, ambiguity, expiry and no forced remap passed\n";
}
