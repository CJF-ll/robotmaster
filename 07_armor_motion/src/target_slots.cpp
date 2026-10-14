#include "target_slots.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace outpost {
void TargetSlots::reconfigure(SlotConfig c){
  sequence_=SlotSequence(c.count,c.sequence_transitions);
  config_=c;profiles_.clear();locals_.clear();last_observation_=-1;++epoch_;
}
TargetSlots::Feature TargetSlots::feature(const Detection& d){
  auto a=d.measurement_points[3]-d.measurement_points[0];
  auto b=d.measurement_points[2]-d.measurement_points[1];
  auto direction=a+b;
  return {d.center.x,d.center.y,(cv::norm(a)+cv::norm(b))/2,
          std::atan2(direction.x,direction.y)};
}
double TargetSlots::cost(const Profile& p,const Feature& f)const{
  if(p.samples.empty()||f.length<=0)return std::numeric_limits<double>::infinity();
  // Compare near the same image x where possible; this accommodates the
  // observed slope of each staggered plate without encoding a trajectory.
  std::vector<std::pair<double,Feature>> neighbors;
  for(auto old:p.samples)neighbors.push_back({std::abs(old.x-f.x),old});
  std::sort(neighbors.begin(),neighbors.end(),[](const auto& a,const auto& b){return a.first<b.first;});
  const size_t n=std::min(size_t(7),neighbors.size());
  std::vector<double> ys,lengths,angles;
  for(size_t i=0;i<n;++i){ys.push_back(neighbors[i].second.y);
    lengths.push_back(neighbors[i].second.length);angles.push_back(neighbors[i].second.angle);}
  auto median=[](std::vector<double>& v){std::sort(v.begin(),v.end());return v[v.size()/2];};
  const double y=median(ys),length=median(lengths),angle=median(angles);
  if(length<=0||std::max(length,f.length)/std::min(length,f.length)>config_.length_ratio)
    return std::numeric_limits<double>::infinity();
  const double angle_delta=std::abs(std::remainder(f.angle-angle,2*CV_PI))*180/CV_PI;
  if(angle_delta>config_.angle_degrees)return std::numeric_limits<double>::infinity();
  const double dy=std::abs(f.y-y)/((length+f.length)/2);
  if(dy>config_.height_gate_lengths)return std::numeric_limits<double>::infinity();
  return dy+.2*std::abs(std::log(f.length/length))+.1*angle_delta/config_.angle_degrees;
}
std::vector<SlotAssignment> TargetSlots::update(double time,const std::vector<TrackOutput>& tracks,
                                               const std::vector<Detection>& detections){
  if(!std::isfinite(time)||(last_time_>=0&&time<=last_time_))throw std::runtime_error("invalid slot time");
  const double previous_time=last_time_;last_time_=time;
  std::vector<SlotAssignment> result(tracks.size());if(!config_.enabled)return result;
  if(last_observation_>=0&&time-last_observation_>config_.reset_gap_seconds)reconfigure(config_);
  for(auto it=locals_.begin();it!=locals_.end();){
    if(time-it->second.last_seen>config_.reset_gap_seconds)it=locals_.erase(it);else ++it;
  }
  struct Proposal {size_t track;int slot;Feature value;};std::vector<Proposal> proposals;
  // Stage new profile hypotheses for same-frame matching. They must not
  // consume persistent IDs or training samples until conflicts are resolved.
  const size_t existing_profiles=profiles_.size();
  auto staged_profiles=profiles_;
  for(size_t i=0;i<tracks.size();++i){const auto& t=tracks[i];
    auto found=locals_.find(t.id);
    if(!t.measurement_used){if(found!=locals_.end()){
      result[i]={found->second.slot,consistent()};found->second.clear_pending();found->second.observations=0;
    }continue;}
    // A new accepted detector measurement must revalidate the current slot.
    // Preserve locked_slot as identity history, but never coast on a rejected
    // or conflicting assignment from an earlier observation.
    const int previous_slot=found==locals_.end()?-1:found->second.slot;
    if(found!=locals_.end())found->second.slot=-1;
    if(t.detection_index<0||size_t(t.detection_index)>=detections.size()){
      if(found!=locals_.end()){found->second.clear_pending();found->second.observations=0;}continue;
    }
    auto& local=locals_[t.id];
    if(local.last_seen!=previous_time){local.clear_pending();local.observations=0;}
    local.last_seen=time;
    const Feature f=feature(detections[t.detection_index]);
    if(!std::isfinite(f.x)||!std::isfinite(f.y)||!std::isfinite(f.length)||!std::isfinite(f.angle)||f.length<=0)
      {local.clear_pending();local.observations=0;continue;}
    ++local.observations;
    std::vector<std::pair<double,int>> costs;
    for(size_t s=0;s<staged_profiles.size();++s){double c=cost(staged_profiles[s],f);if(std::isfinite(c))costs.push_back({c,int(s)});}
    std::sort(costs.begin(),costs.end());int slot=-1;
    if(costs.size()>1&&costs[1].first-costs[0].first<config_.ambiguity_margin){local.clear_pending();continue;}
    if(!costs.empty())slot=costs[0].second;
    else if(local.locked_slot<0&&local.observations>=config_.local_observations&&int(staged_profiles.size())<config_.count){
      slot=int(staged_profiles.size());Profile p;p.samples.push_back(f);staged_profiles.push_back(p);}
    // Never relabel an existing local track merely because its image height
    // drifts toward another profile. Require a new local association instead.
    if(local.locked_slot>=0&&slot!=local.locked_slot){local.clear_pending();continue;}
    if(slot>=0&&size_t(slot)<existing_profiles&&previous_slot!=slot){
      if(local.pending_slot!=slot){local.pending_slot=slot;local.pending_hits=0;}
      ++local.pending_hits;
      if(local.pending_hits<config_.local_observations)continue;
    }else local.clear_pending();
    if(slot>=0)proposals.push_back({i,slot,f});
  }
  std::vector<int> counts(config_.count,0);for(auto p:proposals)++counts[p.slot];
  std::vector<int> committed(config_.count,-1);
  for(size_t s=0;s<existing_profiles;++s)committed[s]=int(s);
  for(const auto& p:proposals)if(counts[p.slot]==1&&size_t(p.slot)>=existing_profiles){
    committed[p.slot]=int(profiles_.size());profiles_.push_back(Profile{});
  }
  int observed_slot=-1;
  for(auto p:proposals){if(counts[p.slot]!=1){locals_[tracks[p.track].id].clear_pending();continue;}
    p.slot=committed[p.slot];
    last_observation_=time;
    auto& local=locals_[tracks[p.track].id];local.slot=p.slot;local.locked_slot=p.slot;
    local.clear_pending();
    auto& samples=profiles_[p.slot].samples;samples.push_back(p.value);
    while(int(samples.size())>config_.profile_samples)samples.pop_front();
    result[p.track]={p.slot,consistent()};
    if(observed_slot==-1)observed_slot=p.slot;else observed_slot=-2;
  }
  // Simultaneous observations do not have an artificial temporal ordering.
  // Enforce uniqueness across observations AND coasting hypotheses. A unique
  // accepted observation supersedes stale coast labels; without such evidence,
  // all competing coast labels remain unknown until re-observed.
  std::vector<int> claims(config_.count,0),observed_claims(config_.count,0);
  for(size_t i=0;i<result.size();++i)if(result[i].slot>=0){
    ++claims[result[i].slot];observed_claims[result[i].slot]+=tracks[i].measurement_used;
  }
  for(size_t i=0;i<result.size();++i){const int slot=result[i].slot;
    if(slot<0||claims[slot]<=1)continue;
    if(observed_claims[slot]==1&&tracks[i].measurement_used)continue;
    locals_[tracks[i].id].slot=-1;result[i]={};
  }
  const auto measurements=std::count_if(tracks.begin(),tracks.end(),[](const auto& t){return t.measurement_used;});
  // Unknown or simultaneous measurements cannot establish temporal adjacency.
  // Missing evidence invalidates the active sequence, not the learned identities.
  if(observed_slot>=0&&measurements==1)sequence_.observe(observed_slot,time);
  else sequence_.interrupt();
  for(auto& value:result)value.sequence_consistent=consistent()&&value.slot>=0;
  return result;
}
}
