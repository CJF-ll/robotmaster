#include "tracker.hpp"
#include "association.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
namespace outpost {
void Tracker::Axis::predict(double dt,double q) {
  position+=velocity*dt;
  p00+=2*dt*p01+dt*dt*p11+q*dt*dt*dt/3;
  p01+=dt*p11+q*dt*dt/2;
  p11+=q*dt;
}
void Tracker::Axis::update(double z,double r) {
  const double s=p00+r,k0=p00/s,k1=p01/s;
  const double residual=z-position;
  position+=k0*residual; velocity+=k1*residual;
  // Joseph form preserves covariance under finite precision.
  const double a=p00,b=p01,c=p11;
  p00=(1-k0)*(1-k0)*a+k0*k0*r;
  p01=(1-k0)*(b-k1*a)+k0*k1*r;
  p11=c-2*k1*b+k1*k1*a+k1*k1*r;
}
static bool valid(const std::array<cv::Point2f,4>& p) {
  for(auto v:p)if(!std::isfinite(v.x)||!std::isfinite(v.y))return false;
  std::vector<cv::Point2f> poly(p.begin(),p.end());
  return cv::isContourConvex(poly)&&cv::contourArea(poly)>4&&
         p[0].x<p[1].x&&p[3].x<p[2].x;
}
static bool common_endpoint_drop(const std::array<cv::Point2f,4>& previous,
                                const Detection& measurement,const TrackerConfig& config){
  const auto& now=measurement.measurement_points;
  const auto old_width=cv::norm((previous[0]+previous[3]-previous[1]-previous[2])*.5f);
  const auto new_width=cv::norm((now[0]+now[3]-now[1]-now[2])*.5f);
  if(old_width<=0||std::abs(new_width/old_width-1)>config.common_endpoint_width_tolerance)return false;
  for(auto edge:{std::pair<int,int>{0,3},{1,2}}){
    const auto old=previous[edge.second]-previous[edge.first];
    const auto current=now[edge.second]-now[edge.first];
    const double old_length=cv::norm(old),length=cv::norm(current);
    if(old_length<=0||length<=0||length/old_length>=config.common_endpoint_shrink_ratio)return false;
    const double angle=std::acos(std::clamp(double(old.dot(current))/(old_length*length),-1.,1.))*180/CV_PI;
    if(angle>config.association_angle_degrees)return false;
  }
  return true;
}
std::vector<TrackOutput> Tracker::step(double time,const std::vector<Detection>& detections,
                                     double horizon) {
  if(!std::isfinite(time)||time<0||!std::isfinite(horizon)||horizon<0||
     (last_time_>=0&&time<=last_time_))throw std::runtime_error("invalid tracker timestamp/horizon");
  const double dt=last_time_<0?0:time-last_time_;last_time_=time;
  if(!std::isfinite(config_.max_shape_drift_px)||config_.max_shape_drift_px<=0||
     config_.shape_history_samples<3||config_.shape_history_samples>30)
    throw std::runtime_error("invalid shape guard settings");
  if(config_.common_endpoint_guard_enabled&&
     (!std::isfinite(config_.common_endpoint_shrink_ratio)||config_.common_endpoint_shrink_ratio<=0||config_.common_endpoint_shrink_ratio>=1||
      !std::isfinite(config_.common_endpoint_width_tolerance)||config_.common_endpoint_width_tolerance<0||config_.common_endpoint_width_tolerance>=.5||
      !std::isfinite(config_.common_endpoint_max_gap_seconds)||config_.common_endpoint_max_gap_seconds<=0||
      config_.common_endpoint_max_gap_seconds>config_.coast_seconds||config_.common_endpoint_min_hits<2||config_.common_endpoint_min_hits>100))
    throw std::runtime_error("invalid common endpoint guard configuration");
  diagnostics_.clear();
  // Validate the actual measurement before association AND birth. A finite
  // display quad alone does not guarantee a usable center/endpoint order.
  std::vector<const char*> invalid(detections.size(),nullptr);
  for(size_t j=0;j<detections.size();++j){const auto& d=detections[j];
    if(!std::isfinite(d.center.x)||!std::isfinite(d.center.y))invalid[j]="nonfinite_center";
    else if(!valid(d.corners))invalid[j]="invalid_display_geometry";
    else if(!valid(d.measurement_points)||d.measurement_points[0].y>=d.measurement_points[3].y||
            d.measurement_points[1].y>=d.measurement_points[2].y)invalid[j]="invalid_measurement_geometry";
    for(auto evidence:d.endpoint_evidence)if(evidence==EndpointEvidence::BoundaryUnsupported)invalid[j]="boundary_endpoint_support_missing";
    if(config_.measurement_quality_enabled&&d.quality_available){
      if(!std::isfinite(d.fit_residual_px)||d.fit_residual_px<0)invalid[j]="invalid_measurement_quality";
      for(double scale:d.center_variance_scale)
        if(!std::isfinite(scale)||scale<1)invalid[j]="invalid_measurement_quality";
    }
  }
  tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const auto& t){
    return time-t.last_observed>config_.coast_seconds;
  }),tracks_.end());
  for(auto& t:tracks_)for(auto& a:t.axes)a.predict(dt,config_.acceleration_noise);
  auto corners=[](const Track& t,double h){
    std::array<cv::Point2f,4> p;
    const cv::Point2f c(float(t.axes[0].position+h*t.axes[0].velocity),
                        float(t.axes[1].position+h*t.axes[1].velocity));
    for(int i=0;i<4;++i)p[i]=c+t.shape[i];
    return p;
  };
  std::vector<std::vector<double>> costs(tracks_.size(),std::vector<double>(
      detections.size(),std::numeric_limits<double>::infinity()));
  auto measurement_variance=[&](const Detection& d){
    const double base=config_.measurement_noise*(d.low_resolution?config_.low_resolution_noise_scale:1.);
    std::array<double,2> r{base,base};
    if(config_.measurement_quality_enabled&&d.quality_available)
      for(int a=0;a<2;++a)r[a]*=std::clamp(d.center_variance_scale[a],1.,config_.measurement_quality_max_scale);
    return r;
  };
  std::vector<std::vector<double>> innovations=costs;
  std::vector<bool> endpoint_suspect(tracks_.size(),false),suspect_candidate(detections.size(),false);
  for(size_t i=0;i<tracks_.size();++i)for(size_t j=0;j<detections.size();++j){
    diagnostics_.push_back({tracks_[i].id,int(j)});
    auto& diagnostic=diagnostics_.back();
    if(invalid[j]){diagnostic.reason=invalid[j];continue;}
    const auto& p=tracks_[i].measurement_shape;if(!valid(p))continue;
    const auto delta=detections[j].center-
        cv::Point2f(float(tracks_[i].axes[0].position),float(tracks_[i].axes[1].position));
    diagnostic.reason="distance_gate";
    if(cv::norm(delta)>config_.association_pixels)continue;
    // Do this BEFORE area/length/NIS gates: severe common truncation must not
    // fail ordinary association and then create a fresh high-confidence track.
    if(config_.common_endpoint_guard_enabled&&tracks_[i].confirmed&&tracks_[i].common_endpoint_guard_ready&&
       (tracks_[i].common_endpoint_suspect_active||time-tracks_[i].last_observed<=config_.common_endpoint_max_gap_seconds+1e-9)&&
       common_endpoint_drop(p,detections[j],config_)){
      diagnostic.reason="suspected_common_endpoint_truncation";
      endpoint_suspect[i]=true;tracks_[i].common_endpoint_suspect_active=true;suspect_candidate[j]=true;continue;
    }
    const auto area=[](const auto& q){return cv::contourArea(std::vector<cv::Point2f>(q.begin(),q.end()));};
    const double ratio=area(p)/area(detections[j].measurement_points);
    diagnostic.reason="measurement_area_gate";
    if(ratio>config_.max_shape_ratio||ratio<1/config_.max_shape_ratio)continue;
    bool geometry_ok=true;
    for(auto edge:{std::pair<int,int>{0,3},{1,2}}){
      const auto old=tracks_[i].measurement_shape[edge.second]-tracks_[i].measurement_shape[edge.first];
      const auto now=detections[j].measurement_points[edge.second]-detections[j].measurement_points[edge.first];
      const double a=cv::norm(old),b=cv::norm(now);
      if(a<=0||b<=0){geometry_ok=false;break;}
      const double angle=std::acos(std::clamp(double(old.dot(now))/(a*b),-1.,1.))*180/CV_PI;
      if(angle>config_.association_angle_degrees||std::max(a,b)/std::min(a,b)>config_.association_length_ratio)
        geometry_ok=false;
    }
    diagnostic.reason="lamp_geometry_gate";
    if(!geometry_ok)continue;
    // Four corners share the same segmented lamps. Their center measurement
    // error is correlated: do not pretend averaging divides that error by 4.
    const auto r=measurement_variance(detections[j]);
    const double sx=r[0]+tracks_[i].axes[0].p00;
    const double sy=r[1]+tracks_[i].axes[1].p00;
    const double nis=delta.x*delta.x/sx+delta.y*delta.y/sy;
    diagnostic.nis=nis;diagnostic.innovation_available=true;
    diagnostic.reason="innovation_gate";
    if(nis<=config_.innovation_gate){
      diagnostic.reason="eligible_not_selected";
      innovations[i][j]=nis;
      // Variance normalization prevents uncertain candidates winning solely
      // because a larger R reduces NIS. At scale=1 this is exactly old NIS.
      costs[i][j]=nis+std::log(sx/(config_.measurement_noise+tracks_[i].axes[0].p00))+
                       std::log(sy/(config_.measurement_noise+tracks_[i].axes[1].p00));
    }
  }
  for(size_t j=0;j<detections.size();++j)if(suspect_candidate[j]){
    bool other_owner=false;for(const auto& row:costs)other_owner|=std::isfinite(row[j]);
    // Another eligible owner may legitimately have shorter lamps; do not
    // globally veto it merely because this candidate is near an older track.
    if(!other_owner)invalid[j]="suspected_common_endpoint_truncation";
  }
  const auto association=associate(costs,detections.size(),config_.association_miss_cost,config_.ambiguity_margin);
  auto associated=association.assignment;
  const auto& ambiguous_detection=association.ambiguous_detections;
  for(size_t i=0;i<tracks_.size();++i){
    if(detections.empty())diagnostics_.push_back({tracks_[i].id,-1,"no_candidates"});
    for(size_t j=0;j<detections.size();++j){auto& d=diagnostics_[i*detections.size()+j];
      if(associated[i]==int(j))d.reason="accepted";
      else if(std::isfinite(costs[i][j])&&association.ambiguous_tracks[i])d.reason="ambiguous_assignment";
    }
  }
  std::vector<bool> used(detections.size(),false);
  std::vector<double> nis(tracks_.size(),0);
  for(size_t i=0;i<associated.size();++i)if(associated[i]>=0){
    used[associated[i]]=true;nis[i]=innovations[i][associated[i]];
    auto& track=tracks_[i];
    if(config_.one_sided_guard_enabled&&track.confirmed&&track.one_sided_guard_ready){
      const auto& before=track.measurement_shape;
      const auto& now=detections[associated[i]].measurement_points;
      const double left=cv::norm(now[3]-now[0])/cv::norm(before[3]-before[0]);
      const double right=cv::norm(now[2]-now[1])/cv::norm(before[2]-before[1]);
      const bool suspect=(left<config_.one_sided_shrink_ratio&&std::abs(right-1)<=config_.other_side_tolerance)||
                         (right<config_.one_sided_shrink_ratio&&std::abs(left-1)<=config_.other_side_tolerance);
      if(suspect){
        diagnostics_[i*detections.size()+size_t(associated[i])].reason="one_sided_length_drop";
        // Reserve the rejected candidate so it cannot immediately bypass the
        // quality gate by spawning a duplicate track. Do not update the shape,
        // position, velocity or last-observed time; ordinary coasting expires it.
        associated[i]=-1;nis[i]=0;track.prediction_ready=false;
      }
    }
  }
  for(size_t i=0;i<tracks_.size();++i){auto& t=tracks_[i];
    t.clean_recovery_used=false;
    if(associated[i]<0){
      if(endpoint_suspect[i])t.prediction_ready=false;
      const bool clean=config_.clean_recovery_enabled&&detections.empty()&&
        !association.ambiguous_tracks[i]&&!t.identity_ambiguous&&
        time-t.last_observed<=config_.clean_recovery_max_gap_seconds+1e-9;
      if(t.consecutive>0){
        // Start eligibility only from a mature track. An interruption while
        // collecting recovery hits has prediction_ready=false and cancels it.
        t.clean_recovery_pending=clean&&t.confirmed&&t.prediction_ready;
      }else t.clean_recovery_pending=t.clean_recovery_pending&&clean;
      if(association.ambiguous_tracks[i])t.identity_ambiguous=true;
      t.consecutive=0;t.health.clear();continue;
    }
    const auto& p=detections[associated[i]].corners;
    const auto c=detections[associated[i]].center;
    t.common_endpoint_suspect_active=false;
    if(t.clean_recovery_pending){
      const auto j=size_t(associated[i]);
      size_t row_options=0,column_options=0;
      for(double cost:costs[i])row_options+=std::isfinite(cost);
      for(const auto& row:costs)column_options+=std::isfinite(row[j]);
      // Check the old timestamp before updating it. Large R can make NIS small,
      // so low-resolution observations cannot qualify for this fast path.
      t.clean_recovery_pending=config_.clean_recovery_enabled&&!t.identity_ambiguous&&
        time-t.last_observed<=config_.clean_recovery_max_gap_seconds+1e-9&&
        row_options==1&&column_options==1&&!detections[j].low_resolution&&
        (!config_.measurement_quality_enabled||!detections[j].quality_available||
         (detections[j].center_variance_scale[0]<=1&&detections[j].center_variance_scale[1]<=1))&&
        std::isfinite(nis[i])&&nis[i]<=config_.clean_recovery_max_nis;
    }
    if(t.consecutive==0)t.prediction_ready=false;
    const auto r=measurement_variance(detections[associated[i]]);
    t.health.add(time,{c.x-t.axes[0].position,c.y-t.axes[1].position},
                 {r[0]+t.axes[0].p00,r[1]+t.axes[1].p00},config_.health);
    t.axes[0].update(c.x,r[0]);
    t.axes[1].update(c.y,r[1]);
    // Preserve the last real convex silhouette. Independently extrapolating
    // corner velocities can invert a narrowing plate and kill association.
    for(int k=0;k<4;++k){t.shape[k]=p[k]-c;
      t.measurement_shape[k]=detections[associated[i]].measurement_points[k]-c;}
    if(time-t.last_observed>dt+1e-9)t.shape_history.clear();
    t.shape_history.push_back({time,t.measurement_shape});
    while(t.shape_history.size()>size_t(config_.shape_history_samples))t.shape_history.pop_front();
    t.last_observed=time;++t.hits;++t.consecutive;
    if(t.consecutive>=config_.one_sided_min_hits)t.one_sided_guard_ready=true;
    if(t.consecutive>=config_.common_endpoint_min_hits)t.common_endpoint_guard_ready=true;
    if(t.consecutive>=config_.confirmation_hits)t.confirmed=true;
    if(t.consecutive>=config_.prediction_hits){
      t.prediction_ready=true;t.identity_ambiguous=false;t.clean_recovery_pending=false;
    }else if(t.clean_recovery_pending&&t.consecutive>=config_.clean_recovery_hits){
      t.prediction_ready=true;t.clean_recovery_used=true;t.clean_recovery_pending=false;
    }
  }
  for(size_t j=0;j<detections.size();++j)if(!used[j]&&!ambiguous_detection[j]&&!invalid[j]){
    Track t{};t.id=next_id_++;t.last_observed=time;t.confirmed=config_.confirmation_hits<=1;
    const auto c=detections[j].center;
    t.axes[0].position=c.x;t.axes[1].position=c.y;
    for(int k=0;k<4;++k){t.shape[k]=detections[j].corners[k]-c;
      t.measurement_shape[k]=detections[j].measurement_points[k]-c;}
    t.shape_history.push_back({time,t.measurement_shape});
    t.prediction_ready=config_.prediction_hits<=1;
    const auto r=measurement_variance(detections[j]);
    for(int a=0;a<2;++a){t.axes[a].p00=r[a];t.axes[a].p11=config_.initial_velocity_variance;}
    tracks_.push_back(t);associated.push_back(int(j));nis.push_back(0);endpoint_suspect.push_back(false);
    diagnostics_.push_back({t.id,int(j),"birth"});
  }
  // id=0 denotes rejection before any identity assignment, also when no track
  // exists yet. Keep original detection indices for downstream diagnostics.
  for(size_t j=0;j<detections.size();++j)if(invalid[j])diagnostics_.push_back({0,int(j),invalid[j]});
  std::vector<TrackOutput> output;
  for(size_t i=0;i<tracks_.size();++i){const auto& t=tracks_[i];if(!t.confirmed)continue;
    TrackOutput o;o.id=t.id;o.measurement_used=associated[i]>=0;o.innovation=nis[i];
    o.health=t.health.summary(config_.health);
    o.detection_index=associated[i];
    if(o.measurement_used){o.measurement_low_resolution=detections[associated[i]].low_resolution;
      o.measurement_variances=measurement_variance(detections[associated[i]]);
      // Retained scalar column is the conservative maximum; new consumers use both axes.
      o.measurement_variance=std::max(o.measurement_variances[0],o.measurement_variances[1]);}
    o.consecutive_hits=t.consecutive;
    o.clean_recovery_active=t.clean_recovery_pending;o.clean_recovery_used=t.clean_recovery_used;
    o.endpoint_truncation_suspected=!o.measurement_used&&t.common_endpoint_suspect_active;
    o.filtered_center={t.axes[0].position,t.axes[1].position};
    o.velocity={t.axes[0].velocity,t.axes[1].velocity};
    for(int axis=0;axis<2;++axis){const auto& a=t.axes[axis];
      o.velocity_variance[axis]=a.p11;
      o.future_position_variance[axis]=a.p00+2*horizon*a.p01+
        horizon*horizon*a.p11+config_.acceleration_noise*std::pow(horizon,3)/3;
    }
    o.current_center=o.measurement_used?detections[associated[i]].center:
                       cv::Point2f(float(t.axes[0].position),float(t.axes[1].position));
    o.future_center={float(t.axes[0].position+horizon*t.axes[0].velocity),
                     float(t.axes[1].position+horizon*t.axes[1].velocity)};
    o.age_seconds=time-t.last_observed;
    o.position_std=std::sqrt(std::max(t.axes[0].p00,t.axes[1].p00));
    o.state=o.measurement_used?TrackState::Tracking:TrackState::Coasting;
    o.current=o.measurement_used?detections[associated[i]].corners:corners(t,0);
    for(int k=0;k<4;++k)o.lamp_support_points[k]=o.measurement_used?
      detections[associated[i]].measurement_points[k]:o.current_center+t.measurement_shape[k];
    o.future=corners(t,horizon);
    o.forecast=assess_forecast(t.prediction_ready,
                              std::isfinite(o.future_center.x)&&std::isfinite(o.future_center.y),o.future_position_variance,
                              o.age_seconds,horizon,config_.coast_seconds,config_.max_position_std,
                              !t.identity_ambiguous);
    o.future_center_valid=o.forecast.usable();
    // Silhouette validity must not revoke a valid center forecast.
    if(!valid(o.future))o.forecast.blocks|=ForecastGeometry;
    // Measure shape change; do not extrapolate corners or claim a calibrated
    // probability. Translation is removed before comparing observed endpoints.
    if(t.shape_history.size()>=size_t(config_.shape_history_samples)){
      double speed=0;
      for(size_t sample=1;sample<t.shape_history.size();++sample){
        const auto& a=t.shape_history[sample-1];const auto& b=t.shape_history[sample];
        for(int k=0;k<4;++k)speed=std::max(speed,cv::norm(b.offsets[k]-a.offsets[k])/(b.time-a.time));
      }
      o.shape_drift_px=speed*(o.age_seconds+horizon);
      o.future_shape_reliable=std::isfinite(o.shape_drift_px)&&o.shape_drift_px<=config_.max_shape_drift_px;
    }
    if(config_.shape_guard_enabled&&!o.future_shape_reliable)o.forecast.blocks|=ForecastShape;
    o.future_valid=o.forecast.usable();
    if(std::isfinite(o.current_center.x)&&std::isfinite(o.current_center.y)&&
       o.position_std<=config_.max_position_std)output.push_back(o);
  }
  return output;
}
}
