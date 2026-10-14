#include "slot_sequence.hpp"
#include "target_slots.hpp"
#include <iostream>
#include <limits>
using namespace outpost;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void train(SlotSequence& sequence,double& time,double dt=.5){
  for(int i=0;i<12;++i){sequence.observe(i%3,time);time+=dt;}
  check(sequence.consistent(),"complete cycles not confirmed");
  check(std::abs(sequence.recurrence_seconds()-3*dt)<1e-9,"observed recurrence incorrect");
}
Detection plate(int slot){
  const float heights[]={100,160,130,220};const float y=heights[slot];
  Detection d;d.center={100,y};d.measurement_points={cv::Point2f(90,y-8),{110,y-8},{110,y+8},{90,y+8}};
  d.corners=d.measurement_points;return d;
}
TrackOutput track(int id,int index=0,bool measured=true){
  TrackOutput t;t.id=id;t.detection_index=index;t.measurement_used=measured;return t;
}
int main(){try{
  const std::vector<int> forward{1,2,0},reverse{2,0,1};
  {
    SlotSequence sequence(3,6);double time=0;train(sequence,time);
    check(sequence.confirmed_successors()==forward,"wrong forward order");
    sequence.observe(1,time);time+=.5; // 2 -> 1 can be a skip, not proof of reversal.
    check(!sequence.consistent()&&sequence.recurrence_seconds()==0,"single skip retained current sequence certainty");
    check(sequence.confirmed_successors()==forward,"single skip overwrote confirmed history");
    for(int i=0;i<12;++i){sequence.observe((i+2)%3,time);time+=.5;}
    check(sequence.consistent()&&sequence.confirmed_successors()==forward,"normal order did not recover");
  }
  {
    SlotSequence sequence(3,6);double time=0;train(sequence,time);
    const int order[]={1,0,2,1,0,2};
    for(int i=0;i<6;++i){sequence.observe(order[i],time);time+=.7;
      if(i<5)check(!sequence.consistent()&&sequence.confirmed_successors()==forward,
                   "partial reversal overwrote confirmed order");}
    check(sequence.consistent()&&sequence.confirmed_successors()==reverse,"repeated reversed order not learned");
    check(sequence.recurrence_seconds()==0,"old-direction recurrence reused after reversal");
    for(int s:{1,0,2}){sequence.observe(s,time);time+=.7;}
    check(sequence.recurrence_seconds()==0,"recurrence reported before three fresh returns");
    for(int s:{1,0}){sequence.observe(s,time);time+=.7;}
    check(std::abs(sequence.recurrence_seconds()-2.1)<1e-9,"new timing was not measured");
  }
  {
    SlotSequence sequence(3,6);double time=0;train(sequence,time);
    sequence.interrupt();
    check(!sequence.consistent()&&sequence.recurrence_seconds()==0&&sequence.confirmed_successors()==forward,
          "gap did not separate current evidence from history");
    for(int i=0;i<20;++i){sequence.observe(1,time);time+=.1;}
    check(!sequence.consistent(),"repeated views of one board counted as transitions");
    for(int i=0;i<20;++i){sequence.observe(i%2,time);time+=.1;}
    check(!sequence.consistent(),"two-board oscillation confirmed a three-board cycle");
    sequence.interrupt();train(sequence,time,1.);
    check(sequence.recurrence_seconds()==3.,"gap-spanning interval contaminated recurrence");
    const auto history=sequence.confirmed_successors();
    bool rejected=false;try{sequence.observe(9,time);}catch(const std::runtime_error&){rejected=true;}
    check(rejected&&sequence.consistent()&&sequence.confirmed_successors()==history,"invalid slot mutated sequence");
    rejected=false;try{sequence.observe(0,std::numeric_limits<double>::quiet_NaN());}catch(const std::runtime_error&){rejected=true;}
    check(rejected&&sequence.consistent(),"nonfinite time mutated sequence");
  }
  // Real TargetSlots integration: gaps affect order evidence, not board identity.
  for(int scenario=0;scenario<6;++scenario){
    SlotConfig config;config.local_observations=1;TargetSlots slots(config);double time=0;
    for(int i=0;i<12;++i){auto r=slots.update(time,{track(i+1)},{plate(i%3)});time+=.5;
      check(r[0].slot==i%3,"training exchanged staggered identities");}
    check(slots.consistent()&&slots.recurrence_seconds()>0,"integration cycle not trained");
    const auto epoch=slots.epoch();
    if(scenario==0)slots.update(time,{},{});
    if(scenario==1){auto r=slots.update(time,{track(12,0,false)},{});
      check(r[0].slot==2&&!r[0].sequence_consistent,"coast lost identity or asserted current order");}
    if(scenario==2){auto r=slots.update(time,{track(100),track(101,1)},{plate(0),plate(1)});
      check(r[0].slot==0&&r[1].slot==1,"simultaneous observations exchanged identities");}
    if(scenario==3)slots.update(time,{track(100)},{plate(3)});
    if(scenario==4)slots.update(time,{track(100),track(101,1)},{plate(0),plate(3)});
    if(scenario==5)slots.update(time,{track(100),track(101)},{plate(0)});
    time+=.5;
    check(!slots.consistent()&&slots.recurrence_seconds()==0&&slots.epoch()==epoch,
          "incomplete evidence kept current order or erased identity profiles");
    for(int i=0;i<12;++i){auto r=slots.update(time,{track(200+i)},{plate(i%3)});time+=.5;
      check(r[0].slot==i%3,"order interruption changed learned identity");}
    check(slots.consistent()&&std::abs(slots.recurrence_seconds()-1.5)<1e-9,
          "fresh complete cycles did not recover order and recurrence");
  }
  std::cout<<"slot sequence gaps, skipped observations, reversal and identity isolation passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
