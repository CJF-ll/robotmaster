#pragma once
#include "tracker.hpp"
#include <deque>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
namespace outpost {
struct AuditedForecast {
  long frame,target;std::uint64_t id;cv::Point2f origin,predicted;
  std::optional<cv::Point2f> challenger;
  bool uninterrupted=true;
};
class PredictionAudit {
 public:
  // Called once per frame, including frames with no tracks. Completion callbacks
  // run before new forecasts are issued, so no future measurement enters a fit.
  template<class Callback> void observe(long frame,
      const std::map<std::uint64_t,const TrackOutput*>& observed,Callback finish){
    if(frame<=last_frame_)throw std::runtime_error("audit frames must increase");
    const bool skipped=last_frame_>=0&&frame!=last_frame_+1;last_frame_=frame;
    for(auto& p:pending_)if(skipped||!observed.count(p.id))p.uninterrupted=false;
    while(!pending_.empty()&&pending_.front().target<=frame){
      const auto p=pending_.front();pending_.pop_front();auto found=observed.find(p.id);
      if(p.target!=frame)finish(p,"timestamp_skipped",nullptr);
      else if(found==observed.end())finish(p,"no_accepted_target",nullptr);
      else if(!p.uninterrupted)finish(p,"interrupted_history",nullptr);
      else finish(p,"scored",found->second);
    }
  }
  void issue(AuditedForecast p){
    if(p.frame!=last_frame_||p.target<=p.frame||
       (!pending_.empty()&&p.target<pending_.back().target))
      throw std::runtime_error("audit requires future targets in chronological order");
    pending_.push_back(p);
  }
  template<class Callback> void finish(Callback callback){
    for(const auto& p:pending_)callback(p,"end_of_video",nullptr);
    pending_.clear();
  }
 private:
  long last_frame_=-1;
  std::deque<AuditedForecast> pending_;
};
}
