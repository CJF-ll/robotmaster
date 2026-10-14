#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <vector>
namespace outpost {
// Observed encounter-order evidence only, not physical rotation direction.
// Call interrupt when there is no unique, accepted slot observation.
class SlotSequence {
 public:
  SlotSequence(int count,int required):count_(count),required_(required){
    if(count<2||count>8||required<2*count)throw std::runtime_error("invalid slot sequence configuration");
    confirmed_next_.assign(count,-1);interrupt();
  }
  void interrupt(){
    next_.assign(count_,-1);entries_.assign(count_,-1);
    last_slot_=-1;transitions_=0;consistent_=false;recurrence_.clear();
    // Keep the previously confirmed order as history, never active evidence.
  }
  void observe(int slot,double time){
    if(slot<0||slot>=count_||!std::isfinite(time)||(last_time_>=0&&time<=last_time_))
      throw std::runtime_error("invalid slot sequence observation");
    last_time_=time;
    if(slot==last_slot_)return;
    const int previous=last_slot_;
    if(previous>=0){
      // One contradictory jump may be a missed board, not a reversal.
      // Restart the candidate graph without overwriting confirmed history.
      if(next_[previous]>=0&&next_[previous]!=slot)interrupt();
      next_[previous]=slot;
      if(transitions_<required_)++transitions_;
    }
    last_slot_=slot;
    const bool confirmed=transitions_>=required_&&complete_cycle();
    if(confirmed&&!consistent_){
      confirmed_next_=next_;entries_.assign(count_,-1);recurrence_.clear();
    }
    consistent_=confirmed;
    if(consistent_){
      if(entries_[slot]>=0){recurrence_.push_back(time-entries_[slot]);
        if(recurrence_.size()>9)recurrence_.pop_front();}
      entries_[slot]=time;
    }
  }
  bool consistent()const{return consistent_;}
  // Historical successors alone must never be used to claim current direction.
  const std::vector<int>& confirmed_successors()const{return confirmed_next_;}
  double recurrence_seconds()const{
    if(!consistent_||recurrence_.size()<3)return 0;
    auto copy=recurrence_;std::sort(copy.begin(),copy.end());return copy[copy.size()/2];
  }
 private:
  bool complete_cycle()const{
    std::vector<bool> visited(count_,false);int current=0;
    for(int i=0;i<count_;++i){
      if(current<0||current>=count_||visited[current])return false;
      visited[current]=true;current=next_[current];
    }
    return current==0;
  }
  int count_,required_,last_slot_=-1,transitions_=0;
  bool consistent_=false;
  double last_time_=-1;
  std::vector<int> next_,confirmed_next_;
  std::vector<double> entries_;
  std::deque<double> recurrence_;
};
}
