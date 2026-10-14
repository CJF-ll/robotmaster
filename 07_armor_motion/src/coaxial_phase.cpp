#include "coaxial_phase.hpp"
#include "association.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace outpost {
bool coaxial_config_valid(const CoaxialConfig& c){
  for(double x:{c.window_seconds,c.fit_interval_seconds,c.min_omega,c.max_omega,c.omega_step,
      c.max_fit_rms,c.ambiguity_ratio,c.innovation_pixels,c.max_phase_correction,c.coast_seconds,c.stop_omega,c.offset_gain,
      c.association_pixels,c.association_margin})
    if(!std::isfinite(x)||x<=0)return false;
  return c.window_seconds<=30&&c.fit_interval_seconds<=c.window_seconds&&
    c.max_omega>c.min_omega&&c.max_omega<=30&&c.omega_step<=c.min_omega&&
    2*c.max_omega/c.omega_step<=2000&&c.ambiguity_ratio>1&&c.max_phase_correction<CV_PI/3&&
    c.association_margin<c.association_pixels&&c.offset_gain<=1&&c.stop_omega<c.min_omega&&c.min_samples>=15&&c.min_slot_samples>=3&&
    c.max_samples>=c.min_samples&&c.max_samples<=1000&&c.phase_samples>=5&&c.phase_samples<=60;
}
CoaxialPhaseModel::CoaxialPhaseModel(CoaxialConfig c):config_(c){
  if(!coaxial_config_valid(c))throw std::runtime_error("invalid coaxial phase configuration");
}
cv::Point2d CoaxialPhaseModel::Projection::point(int slot,double p)const{
  const double a=p+slot*2*CV_PI/3;return centers.at(size_t(slot))+cosine*std::cos(a)+sine*std::sin(a);
}
cv::Point2d CoaxialPhaseModel::Projection::derivative(int slot,double p)const{
  const double a=p+slot*2*CV_PI/3;return cosine*(-std::sin(a))+sine*std::cos(a);
}
std::optional<CoaxialPhaseModel::Projection> CoaxialPhaseModel::fit(double now)const{
  if(samples_.size()<size_t(config_.min_samples))return {};
  std::array<int,3> counts{};for(const auto& s:samples_)++counts[s.slot];
  for(int n:counts)if(n<config_.min_slot_samples)return {};
  if(samples_.back().time-samples_.front().time<config_.window_seconds*.75)return {};
  auto solve=[&](double omega)->std::optional<Projection>{
    const int n=int(samples_.size());cv::Mat x(n,5,CV_64F),y(n,2,CV_64F);
    for(int i=0;i<n;++i){const auto& s=samples_[size_t(i)];
      const double a=omega*(s.time-now)+s.slot*2*CV_PI/3;
      for(int j=0;j<3;++j)x.at<double>(i,j)=s.slot==j?1:0;
      x.at<double>(i,3)=std::cos(a);x.at<double>(i,4)=std::sin(a);
      y.at<double>(i,0)=s.center.x;y.at<double>(i,1)=s.center.y;
    }
    cv::SVD svd(x,cv::SVD::NO_UV);
    if(svd.w.at<double>(4)<svd.w.at<double>(0)*1e-5)return {};
    cv::Mat coefficients;
    if(!cv::solve(x,y,coefficients,cv::DECOMP_SVD))return {};
    Projection p;for(int s=0;s<3;++s)p.centers[s]={coefficients.at<double>(s,0),coefficients.at<double>(s,1)};
    p.cosine={coefficients.at<double>(3,0),coefficients.at<double>(3,1)};
    p.sine={coefficients.at<double>(4,0),coefficients.at<double>(4,1)};
    // Canonical phase zero: maximum horizontal displacement in the fitted
    // affine circle. This fixes the otherwise arbitrary phase/basis gauge.
    const double alpha=std::atan2(p.sine.x,p.cosine.x);
    const auto u=p.cosine,v=p.sine;
    p.cosine=u*std::cos(alpha)+v*std::sin(alpha);
    p.sine=v*std::cos(alpha)-u*std::sin(alpha);
    const double extent=std::hypot(cv::norm(u),cv::norm(v));
    if(!std::isfinite(extent)||extent<5||extent>5000)return {};
    p.omega=omega;p.phase=-alpha;
    double sum=0;for(const auto& s:samples_){const auto e=p.point(s.slot,p.phase+omega*(s.time-now))-s.center;sum+=e.dot(e);}
    p.rms=std::sqrt(sum/samples_.size());
    if(!std::isfinite(p.rms))return {};
    return p;
  };
  std::vector<Projection> candidates;
  for(int sign:{-1,1})for(double w=config_.min_omega;w<=config_.max_omega+1e-9;w+=config_.omega_step)
    if(auto p=solve(sign*w))candidates.push_back(*p);
  if(candidates.empty())return {};
  std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.rms<b.rms;});
  auto best=candidates.front();
  const double center=best.omega;
  for(int i=-5;i<=5;++i){const double w=center+i*config_.omega_step/5;
    if(std::abs(w)<config_.min_omega||std::abs(w)>config_.max_omega)continue;
    if(auto p=solve(w);p&&p->rms<best.rms)best=*p;
  }
  if(best.rms>config_.max_fit_rms)return {};
  // Different frequency/sign branches must be observably worse, not merely
  // adjacent grid points around the same minimum.
  for(const auto& p:candidates)if(std::abs(p.omega-best.omega)>.3&&
      p.rms<config_.ambiguity_ratio*std::max(best.rms,.1))return {};
  return best;
}
CoaxialState CoaxialPhaseModel::update(double time,const std::vector<CoaxialObservation>& input,double horizon){
  if(!std::isfinite(time)||time<0||!std::isfinite(horizon)||horizon<0||
     (last_time_>=0&&time<=last_time_))throw std::runtime_error("invalid coaxial timestamp/horizon");
  std::array<bool,3> used{};
  for(const auto& o:input){
    if(o.slot< -1||o.slot>=3||(o.slot>=0&&used[o.slot])||!std::isfinite(o.center.x)||!std::isfinite(o.center.y))
      throw std::runtime_error("invalid/duplicate coaxial observation");
    if(o.slot>=0)used[o.slot]=true;
  }
  const double dt=last_time_<0?0:time-last_time_;last_time_=time;
  if(projection_)phase_+=omega_*dt;
  if(last_observed_>=0&&time-last_observed_>config_.coast_seconds){
    projection_.reset();phases_.clear();
  }
  std::vector<CoaxialObservation> observations;std::vector<int> assignments(input.size(),-1);
  if(projection_){
    std::vector<std::vector<double>> costs(3,std::vector<double>(input.size(),INFINITY));
    for(int s=0;s<3;++s)for(size_t j=0;j<input.size();++j){
      if(input[j].slot>=0&&input[j].slot!=s)continue;
      const double d=cv::norm(input[j].center-projection_->point(s,phase_)-offsets_[s]);
      if(d<=config_.association_pixels)costs[s][j]=d;
    }
    const auto matching=associate(costs,input.size(),config_.association_pixels,config_.association_margin);
    for(int s=0;s<3;++s){const int j=matching.assignment[size_t(s)];
      if(j<0||matching.ambiguous_tracks[size_t(s)]||matching.ambiguous_detections[size_t(j)])continue;
      assignments[size_t(j)]=s;observations.push_back({s,input[size_t(j)].center});
    }
  }else for(size_t j=0;j<input.size();++j)if(input[j].slot>=0){
    observations.push_back(input[j]);assignments[j]=input[j].slot;
  }
  // Bootstrap/rebootstrap uses independent labeled tracks only. Do not train
  // the geometry on slot guesses produced by that same projection model.
  for(const auto& o:input)if(o.slot>=0)samples_.push_back({time,o.slot,o.center});
  while(!samples_.empty()&&(time-samples_.front().time>config_.window_seconds||
        samples_.size()>size_t(config_.max_samples)))samples_.pop_front();
  bool fresh_fit=false;
  if(!projection_&&!observations.empty()&&(last_fit_<0||time-last_fit_>=config_.fit_interval_seconds)){
    last_fit_=time;
    // Fixed camera and fixed common axis: keep the learned phase basis stable.
    // Refit only after loss/reset, never reinterpret the phase every few frames.
    if(auto fitted=fit(time)){
      offsets_={};++generation_;projection_=*fitted;phase_=fitted->phase;omega_=fitted->omega;
      omega_variance_=config_.omega_step*config_.omega_step;
      phases_.clear();fresh_fit=true;
    }
  }
  CoaxialState out;out.generation=generation_;out.initialized=bool(projection_);
  out.observation_slots.assign(input.size(),-1);
  if(!projection_)return out;
  double corrected=phase_,error=0,information=0;
  if(!observations.empty()){
    for(int iteration=0;iteration<8;++iteration){double numerator=0;information=0;
      for(const auto& o:observations){const auto d=projection_->derivative(o.slot,corrected);
        const auto e=o.center-projection_->point(o.slot,corrected)-offsets_[o.slot];numerator+=d.dot(e);information+=d.dot(d);}
      if(information<1e-6)break;
      const double change=std::clamp(numerator/information,-.1,.1);corrected+=change;
      if(std::abs(change)<1e-7)break;
    }
    for(const auto& o:observations){const auto e=o.center-projection_->point(o.slot,corrected)-offsets_[o.slot];error+=e.dot(e);}
    error=std::sqrt(error/observations.size());
    if(information>1&&error<=config_.innovation_pixels&&std::abs(corrected-phase_)<=config_.max_phase_correction){
      phase_=corrected;last_observed_=time;out.measurement_used=true;
      out.observation_slots=assignments;
      for(const auto& o:observations)offsets_[o.slot]+=
        (o.center-projection_->point(o.slot,phase_)-offsets_[o.slot])*config_.offset_gain;
      phase_variance_=std::pow(std::max(2.,projection_->rms),2)/information;
      phases_.push_back({time,phase_});while(phases_.size()>size_t(config_.phase_samples))phases_.pop_front();
      if(phases_.size()>=5&&!fresh_fit){
        double mt=0,mp=0,tt=0,tp=0;
        for(auto s:phases_){mt+=(s.first-time)/phases_.size();mp+=s.second/phases_.size();}
        for(auto s:phases_){const double t=s.first-time-mt;tt+=t*t;tp+=t*(s.second-mp);}
        if(tt>1e-9){const double w=tp/tt;if(std::abs(w)<=config_.max_omega){
          omega_=w;double residual=0;
          for(auto s:phases_){const double e=s.second-mp-w*(s.first-time-mt);residual+=e*e;}
          omega_variance_=std::max(phase_variance_,residual/(phases_.size()-2))/tt;
        }}
      }
    }
  }
  out.phase=phase_;out.omega=omega_;out.fit_rms=projection_->rms;
  out.cosine=projection_->cosine;out.sine=projection_->sine;
  for(int s=0;s<3;++s)out.orbit_centers[s]=projection_->centers[s]+offsets_[s];
  out.age_seconds=last_observed_<0?INFINITY:time-last_observed_;
  out.valid=out.age_seconds<=config_.coast_seconds&&horizon+out.age_seconds<=config_.coast_seconds;
  out.reason=out.measurement_used?"observed":out.valid?"coasting":"rejected_or_expired";
  out.period_seconds=std::abs(omega_)>config_.stop_omega?2*CV_PI/std::abs(omega_):0;
  const double span=horizon+(std::isfinite(out.age_seconds)?out.age_seconds:0);
  const double angular_variance=std::pow(std::sqrt(phase_variance_)+span*std::sqrt(omega_variance_),2);
  for(int s=0;s<3;++s){out.current[s]=projection_->point(s,phase_)+offsets_[s];out.future[s]=projection_->point(s,phase_+omega_*horizon)+offsets_[s];
    const auto d=projection_->derivative(s,phase_+omega_*horizon);
    out.future_variance[s]={projection_->rms*projection_->rms+d.x*d.x*angular_variance,
                           projection_->rms*projection_->rms+d.y*d.y*angular_variance};
  }
  return out;
}
}
