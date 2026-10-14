#include "rigid_estimator.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <iomanip>

using namespace outpost;
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
RigidEstimatorConfig config(bool staggered=true){
  RigidEstimatorConfig c;c.camera_matrix=(cv::Mat_<double>(3,3)<<1100,0,640,0,1100,360,0,0,1);
  c.distortion=(cv::Mat_<double>(1,5)<<-.06,.01,0,0,0);
  c.camera_from_target_rotation=(cv::Mat_<double>(3,3)<<0,1,0,0,0,-1,-1,0,0);
  c.lightbar_length_m=.06;c.lightbar_center_distance_m=.135;c.radius_m=.27;c.pitch_rad=-.15;
  c.slot_heights_m=staggered?std::array<double,3>{{0,.08,.17}}:std::array<double,3>{{0,0,0}};
  c.max_center_error_m=.065;c.identity_score_margin=2;c.history_seconds=.20;
  return c;
}
// Independent generative camera: explicit tangent/up endpoints, no estimator
// state or private projection helper is used to create the observations.
Detection observe(const RigidEstimatorConfig& c,cv::Vec3d center,double phase,int slot,double noise=0,int frame=0){
  const double a=phase+slot*2*CV_PI/3;const cv::Vec3d radial(std::cos(a),std::sin(a),0);
  const cv::Vec3d tangent(-std::sin(a),std::cos(a),0),up=std::sin(c.pitch_rad)*radial+cv::Vec3d(0,0,std::cos(c.pitch_rad));
  const cv::Matx33d basis(c.camera_from_target_rotation);std::vector<cv::Point3d> points;
  for(auto sign:std::array<cv::Point2d,4>{{{-1,1},{1,1},{1,-1},{-1,-1}}}){
    const auto p=basis*(center+c.radius_m*radial+cv::Vec3d(0,0,c.slot_heights_m[slot])+
      sign.x*c.lightbar_center_distance_m*.5*tangent+sign.y*c.lightbar_length_m*.5*up);
    points.push_back({p[0],p[1],p[2]});
  }
  std::vector<cv::Point2d> pixel;cv::projectPoints(points,cv::Vec3d(),cv::Vec3d(),c.camera_matrix,c.distortion,pixel);
  Detection d;d.center={0,0};for(int k=0;k<4;++k){
    pixel[k].x+=noise*std::sin(frame*1.7+k*.8);pixel[k].y+=noise*std::cos(frame*1.3+k*1.1);
    d.measurement_points[k]=pixel[k];d.corners[k]=pixel[k];d.center+=d.measurement_points[k]*.25f;
  }return d;
}
int visible(double phase){int best=0;double error=10;for(int s=0;s<3;++s){const double e=std::abs(std::remainder(phase+s*2*CV_PI/3,2*CV_PI));if(e<error){error=e;best=s;}}return best;}
double phase_error(double a,double b){return std::abs(std::remainder(a-b,2*CV_PI));}
void configuration_contracts(){
  for(int which=0;which<8;++which){auto c=config();
    if(which==0)c.camera_matrix.release();
    if(which==1)c.distortion.release();
    if(which==2)c.camera_from_target_rotation.release();
    if(which==3)c.radius_m=0;
    if(which==4)c.slot_heights_m[1]=NAN;
    if(which==5)c.pitch_rad=NAN;
    if(which==6)c.camera_from_target_rotation.at<double>(0,0)=1;
    if(which==7)c.camera_matrix.at<double>(0,0)=NAN;
    bool rejected=false;try{CalibratedRigidEstimator estimator(c);}catch(const std::exception&){rejected=true;}
    require(rejected,"missing/invalid physical calibration accepted");
  }
}
void staggered_rotation_and_coast(){
  auto c=config();CalibratedRigidEstimator e(c);RigidEstimate r;const cv::Vec3d center(-3,0,.2);
  // First viewed physical slot is 1, not 0. Correct height correspondence must
  // be recovered from later plates rather than arbitrarily anchoring slot 0.
  const double initial=-2*CV_PI/3+.35,omega=1.35;bool saw_initializing=false;int mask=0;
  for(int i=0;i<225;++i){const double t=i/30.,p=initial+omega*t;const int s=visible(p);
    r=e.update(t,{observe(c,center,p,s)},.10);saw_initializing|=!r.identity_resolved;
    if(i<10)require(!r.identity_resolved,"single visible staggered plate falsely fixes unseen height identities");
    if(r.valid){require(std::abs(r.omega-omega)<.02,"rotation omega differs from observed rate");
      require(cv::norm(r.center-center)<.002,"staggered common center wrong");
      require(phase_error(r.phase,p)<.005,"unwrapped phase wrong");
      require(r.slots[s].observed,"fixed slot association switched");mask|=1<<s;
      for(int slot=0;slot<3;++slot){require(r.slots[slot].future_valid,"three future slots not projected");
        const auto truth=observe(c,center,p+omega*.10,slot);
        for(int k=0;k<4;++k)require(cv::norm(truth.measurement_points[k]-r.slots[slot].future_points[k])<.1,"3D future reprojection wrong");}
    }
  }
  require(saw_initializing&&r.valid&&mask==7,"staggered identities never resolved");
  const double last=224/30.;const auto coast=e.update(last+.05,{},.10);
  require(coast.valid&&!coast.measurement_used&&coast.state==RigidTrackingState::Coasting,"short missing observation did not coast");
  require(phase_error(coast.phase,r.phase+r.omega*.05)<1e-8,"coast phase not causal propagation");
  const auto expired=e.update(last+.31,{},.10);require(!expired.initialized&&!expired.valid,"expired rigid target still valid");
  const auto recovered=e.update(last+.34,{observe(c,center,initial+omega*(last+.34),visible(initial+omega*(last+.34)))},.1);
  require(recovered.initialized&&!recovered.motion_confirmed&&!recovered.valid,"new target inherited old speed confidence");
  require(recovered.generation==r.generation+1,"expiry/reacquisition did not change generation");
  const double again=last+.70;
  const auto same_frame=e.update(again,{observe(c,center,initial+omega*again,visible(initial+omega*again))},.1);
  require(same_frame.initialized&&same_frame.generation==recovered.generation+1,"expiry with immediate observation retained old generation");
}
void stationary_reverse_and_multi_observation(){
  auto c=config(false);CalibratedRigidEstimator e(c);const cv::Vec3d center(-3,0,.2);RigidEstimate r;
  double phase=.2;
  for(int i=0;i<180;++i){const double t=i/30.;double omega=i<60?0:(i<120?1.1:-1.1);if(i)phase+=omega/30.;
    const int s=visible(phase);r=e.update(t,{observe(c,center,phase,s)},.1);
    if(i==0)require(!r.motion_confirmed&&!r.valid,"initial zero speed marked known");
    if(i==40)require(r.valid&&std::abs(r.omega)<.01,"observed stationary target failed");
    if(i==110)require(r.valid&&std::abs(r.omega-1.1)<.02,"positive rate not estimated");
    if(i==175)require(r.valid&&std::abs(r.omega+1.1)<.02,"reversal not estimated");
  }
  require(r.relative_identity_only&&r.identity_resolved,"symmetric relative identity not explicit");
  CalibratedRigidEstimator multiple(c);
  for(int i=0;i<15;++i){const double p=CV_PI/3+.01*i;
    r=multiple.update(i/30.,{observe(c,center,p,0),observe(c,center,p,2)},.1);
    require(r.accepted_measurements==2,"same-frame observations not jointly accepted");
    require(r.slots[0].detection_index!=r.slots[2].detection_index,"one detection assigned twice");
  }
  require(r.valid&&std::abs(r.omega-.3)<.03,"multi-observation common motion incorrect");
}
void translation_and_unknown_identity(){
  auto c=config();CalibratedRigidEstimator unknown(c);const cv::Vec3d center(-3,0,.2);
  for(int i=0;i<80;++i){const auto r=unknown.update(i/30.,{observe(c,center,.2-2*CV_PI/3,1)},.1);
    require(!r.identity_resolved&&!r.valid,"repeated single height observation resolves unseen slots");
    if(i>10)for(const auto& slot:r.slots){
      require(slot.future_uncertainty_available&&!slot.future_valid,
              "unresolved identity suppressed diagnostics or allowed drawing");
      require(slot.forecast_block_reason=="identity_unresolved","identity rejection reason missing");}}
  c=config(false);CalibratedRigidEstimator e(c);const cv::Vec3d velocity(.03,.12,.02);RigidEstimate r;
  for(int i=0;i<60;++i){const double t=i/30.,p=.3+.4*t;
    r=e.update(t,{observe(c,center+velocity*t,p,visible(p))},.1);}
  require(r.valid&&cv::norm(r.velocity-velocity)<.002,"common center velocity not estimated from observations");
  const double t=59/30.;const auto truth=observe(c,center+velocity*(t+.1),.3+.4*(t+.1),0);
  for(int k=0;k<4;++k)require(cv::norm(truth.measurement_points[k]-r.slots[0].future_points[k])<.1,"translated future projection incorrect");
  const auto bounded=e.update(t+.01,{observe(c,center+velocity*(t+.01),.3+.4*(t+.01),visible(.3+.4*(t+.01)))},1);
  for(const auto& slot:bounded.slots){require(!slot.future_valid,"prediction beyond configured coast horizon accepted");
    require(slot.future_uncertainty_available&&std::isfinite(slot.max_future_corner_std_px),
            "expired horizon suppressed available uncertainty diagnostics");
    require(slot.forecast_block_reason=="horizon_expired","expired horizon reason missing");}
}
void assignment_ambiguity_contracts(){
  auto c=config(false);CalibratedRigidEstimator a(c),b(c);const cv::Vec3d center(-3,0,.2);
  RigidEstimate r1,r2;
  auto one=[&](int i){return observe(c,center,.2+.5*i/30.,0);};
  for(int i=0;i<20;++i){r1=a.update(i/30.,{one(i)},.1);r2=b.update(i/30.,{one(i)},.1);}
  require(r1.valid&&r2.valid,"ambiguity test did not acquire motion");
  const double t=20/30.,p=.2+.5*t;
  const auto d1=observe(c,center,p-.005,0),d2=observe(c,center,p+.005,0);
  r1=a.update(t,{d1,d2},.1);r2=b.update(t,{d2,d1},.1);
  require(!r1.measurement_used&&!r2.measurement_used&&r1.association_ambiguous&&r2.association_ambiguous,"equal competing ownership was arbitrarily accepted");
  require(!r1.valid&&!r2.valid&&r1.sample_count==r2.sample_count,"ambiguous ownership remained reliable");
  require(phase_error(r1.phase,r2.phase)<1e-12&&cv::norm(r1.center-r2.center)<1e-12,"candidate order changed ambiguous state");
  for(const auto& slot:r1.slots)require(!slot.future_valid,"ambiguous identity produced future projection");
  require(r1.diagnostics[0].reason=="ambiguous_assignment","ambiguous assignment reason missing");
  r1=a.update(21/30.,{},.1);require(r1.association_ambiguous&&!r1.valid,"empty frame cleared ambiguity");
  for(int i=22;i<22+c.min_samples;++i){r1=a.update(i/30.,{one(i)},.1);
    require(r1.association_ambiguous==(i<21+c.min_samples),"ownership recovery ignored required consecutive evidence");}
  require(r1.valid,"clean ownership did not recover after required evidence");
  const int next=22+c.min_samples;
  const auto duplicate=one(next);r1=a.update(next/30.,{duplicate,duplicate},.1);
  require(r1.association_ambiguous&&!r1.measurement_used,"identical duplicate candidates bypassed ownership ambiguity");
  r1=a.update((next+1)/30.,{one(next+1)},.1);
  r1=a.update((next+2)/30.,{},.1);require(r1.association_ambiguous,"recovery gap cleared ambiguity");
  for(int i=next+3;i<next+3+c.min_samples-1;++i){r1=a.update(i/30.,{one(i)},.1);require(r1.association_ambiguous,"recovery counted across missing frame");}
}
void initial_pose_score_contract(){
  auto c=config(false);CalibratedRigidEstimator e(c);Detection d;
  // Free IPPE starts formerly traded pixel error against orientation error.
  // Both now converge to the same constrained geometry; a duplicate minimum
  // must not masquerade as two ambiguous identities. The diagnostic score must
  // describe the actual projected constrained plate, not its free-PnP seed.
  d.measurement_points={{{615.647522f,267.170441f},{667.044434f,267.091949f},
                        {667.857605f,293.222260f},{614.501099f,290.860718f}}};
  d.corners=d.measurement_points;
  const auto r=e.update(0,{d},.1);
  require(r.initialized&&r.measurement_used&&!r.valid,"converged constrained minimum not acquired without inventing initial velocity");
  double squared=0;for(int k=0;k<4;++k){const auto residual=r.slots[0].current_points[k]-d.measurement_points[k];squared+=residual.dot(residual);}
  require(std::abs(std::sqrt(squared/4)-r.diagnostics[0].reprojection_error_px)<1e-4,"pose gate RMS differs from final projected geometry");
}
void hypothesis_expiration_contract(){
  auto c=config();c.identity_score_margin=1e6;
  CalibratedRigidEstimator e(c);const cv::Vec3d center(-3,0,.2);std::uint64_t latest=0;
  for(int i=0;i<225;++i){const double t=i/30.,p=-2*CV_PI/3+.35+1.35*t;
    const auto r=e.update(t,{observe(c,center,p,visible(p))},.1);
    require(!r.identity_resolved&&!r.valid,"expired alternatives were counted as positive identity evidence");
    require(r.generation>=latest,"acquisition generation moved backwards");latest=r.generation;
  }
  // The genuine branch keeps observing. Wrong cyclic-height hypotheses expire
  // individually; the original all-of expiry condition left them revivable
  // forever and never opened a fresh acquisition under this strict margin.
  require(latest>1,"expired individual hypotheses retained stale score/seen state");
}
void noisy_degenerate_and_causality(){
  auto c=config(false);CalibratedRigidEstimator a(c),b(c);cv::Vec3d center(-3,0,.2);RigidEstimate r;
  for(int i=0;i<90;++i){const double p=.15+.65*i/30.;const auto d=observe(c,center,p,visible(p),.10,i);
    r=a.update(i/30.,{d},.1);const auto other=b.update(i/30.,{d},.1);
    require(phase_error(r.phase,other.phase)<1e-12&&std::abs(r.omega-other.omega)<1e-12,"identical prefixes produce different state");
  }
  require(r.valid&&std::abs(r.omega-.65)<.15,"noisy observed rotation not estimated");
  auto invalid=observe(c,center,0,0);invalid.measurement_points.fill({10,10});
  auto out=a.update(3.1,{invalid},.1);require(!out.measurement_used,"degenerate observation used");
  require(out.diagnostics[0].reason=="degenerate_measurement","degenerate reason missing");
  require(std::isnan(out.diagnostics[0].reprojection_error_px)&&std::isnan(out.diagnostics[0].orientation_error_rad),
          "uncomputed pose errors falsely reported as perfect zero fit");
  auto neural=observe(c,center,.1,0);neural.kind=ObservationKind::NeuralKeypoints;
  out=a.update(3.2,{neural},.1);require(!out.measurement_used,"neural corners passed as real centerline measurement");
  bool rejected=false;try{a.update(3.2,{},.1);}catch(const std::exception&){rejected=true;}
  require(rejected,"duplicate timestamp accepted");
  CalibratedRigidEstimator reject(c);
  auto malformed=observe(c,center,.3,0);malformed.measurement_points[0].x+=25;
  out=reject.update(0,{malformed},.1);
  require(!out.measurement_used,"gross nonrigid four-point observation accepted");
  auto flipped=observe(c,center,.3,0);
  std::swap(flipped.measurement_points[0],flipped.measurement_points[3]);
  std::swap(flipped.measurement_points[1],flipped.measurement_points[2]);
  out=reject.update(.1,{flipped},.1);
  require(!out.measurement_used,"upside-down endpoint correspondence accepted");
}
void measurement_boundary_contracts(){
  const auto c=config(false);const cv::Vec3d center(-3,0,.2);
  for(auto kind:{ObservationKind::LampPair,ObservationKind::NeuralSupportedLampPair})for(int side=0;side<2;++side){
    auto d=observe(c,center,.4,0);d.kind=kind;
    d.endpoint_evidence[side]=EndpointEvidence::BoundaryUnsupported;
    CalibratedRigidEstimator fresh(c);const auto first=fresh.update(0,{d},.1);
    require(!first.initialized&&!first.measurement_used,"unsupported lamp boundary initialized rigid target");
    require(first.diagnostics[0].reason=="boundary_endpoint_support_missing",
            "unsupported boundary rejection reason missing");
    require(std::isnan(first.diagnostics[0].reprojection_error_px),"boundary rejection ran pose fitting");
    CalibratedRigidEstimator tracked(c),empty_reference(c);
    for(int i=0;i<15;++i){const auto full=observe(c,center,.4+.3*i/30.,0);
      tracked.update(i/30.,{full},.1);empty_reference.update(i/30.,{full},.1);}
    d=observe(c,center,.4+.3*15/30.,0);d.kind=kind;
    d.endpoint_evidence[side]=EndpointEvidence::BoundaryUnsupported;
    const auto blocked=tracked.update(15/30.,{d},.1),missing=empty_reference.update(15/30.,{},.1);
    require(!blocked.measurement_used&&blocked.state==RigidTrackingState::Coasting,
            "unsupported endpoints updated existing rigid target");
    require(blocked.sample_count==missing.sample_count&&blocked.generation==missing.generation&&
            cv::norm(blocked.center-missing.center)<1e-12&&phase_error(blocked.phase,missing.phase)<1e-12&&
            std::abs(blocked.omega-missing.omega)<1e-12,"rejected boundary contaminated motion history");
    const auto recovered=tracked.update(16/30.,{observe(c,center,.4+.3*16/30.,0)},.1);
    require(recovered.measurement_used&&recovered.generation==blocked.generation,
            "full measured endpoints failed to recover existing identity");
  }
  // Failed optional gray refinement is not the same as a missing boundary.
  for(int malformed=0;malformed<3;++malformed){
    CalibratedRigidEstimator e(c);auto d=observe(c,center,.4,0);
    if(malformed==0)d.measurement_points[0].x=NAN;
    if(malformed==1)d.measurement_points[1].y=INFINITY;
    if(malformed==2){d.measurement_points[1]=d.measurement_points[0];d.measurement_points[2]=d.measurement_points[3];}
    const auto rejected=e.update(0,{d},.1);
    require(!rejected.initialized&&!rejected.measurement_used&&
            rejected.diagnostics[0].reason=="degenerate_measurement","nonfinite or single-lamp geometry passed rigid boundary");
  }
  for(auto evidence:{EndpointEvidence::NotEvaluated,EndpointEvidence::Uncertain,EndpointEvidence::ContrastSupported}){
    CalibratedRigidEstimator e(c);auto d=observe(c,center,.4,0);d.endpoint_evidence={evidence,evidence};
    require(e.update(0,{d},.1).measurement_used,"valid binary measurements rejected by gray-refinement metadata");}
}
void constrained_projection_and_uncertainty(){
  const cv::Vec3d center(-3,0,.2);auto c=config(false);c.history_seconds=.3;
  auto high_noise=c;high_noise.pixel_noise_std_px=2*c.pixel_noise_std_px;
  auto strict=c;strict.max_future_corner_std_px=.001;
  CalibratedRigidEstimator normal(c),larger(high_noise),gated(strict);
  RigidEstimate a,b,g;
  for(int i=0;i<45;++i){const double t=i/30.,p=.35+.3*t;
    const auto d=observe(c,center,p,0,.03,i);a=normal.update(t,{d},.1);b=larger.update(t,{d},.1);g=gated.update(t,{d},.1);}
  require(a.valid&&a.state_reliable&&a.uncertainty_available,"well-conditioned state has no uncertainty estimate");
  require(b.phase_std_rad>a.phase_std_rad*1.7&&b.center_std_m>a.center_std_m*1.7,"pixel-noise assumption did not reach state covariance");
  require(a.omega_std_rad_s>0&&std::isfinite(a.omega_std_rad_s),"velocity uncertainty not reported");
  for(int s=0;s<3;++s){require(a.slots[s].future_valid&&a.slots[s].prediction_reliable,"nominal synthetic prediction incorrectly blocked");
    require(a.slots[s].future_uncertainty_available&&a.slots[s].forecast_block_reason=="none",
            "valid prediction diagnostic inconsistent");
    require(g.slots[s].future_geometry_valid&&!g.slots[s].prediction_reliable&&!g.slots[s].future_valid,"geometry/reliability were conflated");
    require(g.slots[s].future_uncertainty_available&&g.slots[s].forecast_block_reason=="corner_uncertainty_exceeded",
            "uncertainty rejection lacks computed diagnostics/reason");
    require(b.slots[s].max_future_corner_std_px>a.slots[s].max_future_corner_std_px,"pixel noise did not propagate to corners");}
  const auto near=normal.update(45/30.,{},.03),far=normal.update(46/30.,{},.14);
  require(near.valid&&far.valid,"short uncertainty coast unexpectedly expired");
  require(far.slots[0].max_future_corner_std_px>near.slots[0].max_future_corner_std_px,"coast/horizon uncertainty failed to grow");
  // A stricter information floor rejects the observation itself, whereas a
  // future uncertainty limit can leave geometric projection available.
  auto weak=c;weak.min_geometry_information_ratio=.5;CalibratedRigidEstimator ill_conditioned(weak);
  const auto rejected=ill_conditioned.update(0,{observe(c,center,.2,0)},.1);
  require(!rejected.measurement_used&&!rejected.initialized,"ill-conditioned pose bypassed information gate");
  for(int i=0;i<6;++i){auto invalid=c;
    if(i==0)invalid.pixel_noise_std_px=0;if(i==1)invalid.pose_refinement_iterations=0;
    if(i==2)invalid.min_geometry_information_ratio=1;if(i==3)invalid.max_phase_std_rad=CV_PI;
    if(i==4)invalid.angular_acceleration_std_rad_s2=NAN;if(i==5)invalid.motion_change_confirmations=1;
    bool threw=false;try{CalibratedRigidEstimator e(invalid);}catch(const std::exception&){threw=true;}require(threw,"invalid uncertainty/adaptation configuration accepted");}
  // A tilted free pose can fit its input but cannot be used as the RMS for a
  // fixed-axis/pitch rendered pose. Every accepted single-frame fixture must
  // instead report its own constrained projection error.
  for(int i=0;i<25;++i){CalibratedRigidEstimator e(c);const auto d=observe(c,center,-.7+i*.055,0,.5,i);
    const auto r=e.update(0,{d},.1);if(!r.measurement_used)continue;
    double squared=0;for(int k=0;k<4;++k){const auto error=r.slots[0].current_points[k]-d.measurement_points[k];squared+=error.dot(error);}
    require(std::abs(std::sqrt(squared/4)-r.diagnostics[0].reprojection_error_px)<1e-4,"constrained output and reprojection gate differ");
    require(r.diagnostics[0].geometry_information_ratio>=c.min_geometry_information_ratio,"accepted pose has invalid geometry information");}
}
double manoeuvre_phase(double t){
  const double forward=1.2*std::clamp(t-1.,0.,1.5);
  const double reverse=-1.1*std::clamp(t-3.5,0.,1.5);
  const double restart=1.0*std::clamp(t-5.,0.,.7);
  return .45+forward+reverse+restart;
}
double manoeuvre_rate(double t){return t<1?0:t<2.5?1.2:t<3.5?0:t<5?-1.1:t<5.7?1.:0;}
struct TransitionMetrics {
  int frames=0,valid=0,geometric_corners=0,valid_corners=0,adaptive_frames=0;
  double omega_squared=0,corner_squared=0,valid_corner_squared=0;
  std::array<double,5> recovery{{NAN,NAN,NAN,NAN,NAN}};
};
TransitionMetrics score_transitions(bool adaptive,double noise){
  auto c=config(false);c.history_seconds=.3;c.adaptive_window_enabled=adaptive;
  CalibratedRigidEstimator e(c);const cv::Vec3d center(-3,0,.2);TransitionMetrics metrics;
  const std::array<double,5> changes{{1,2.5,3.5,5,5.7}};std::array<int,5> stable{};
  for(int i=0;i<225;++i){const double t=i/30.,p=manoeuvre_phase(t),h=.1;
    // One short, deterministic loss immediately following the stop tests
    // recovery without hiding the difficult transition from the denominator.
    std::vector<Detection> measurements;
    if(i!=80&&i!=81)measurements.push_back(observe(c,center,p,visible(p),noise,i));
    const auto r=e.update(t,measurements,h);
    const double rate_error=std::abs(r.omega-manoeuvre_rate(t));
    for(size_t event=0;event<changes.size();++event)if(t+1e-9>=changes[event]&&t<changes[event]+.65){
      if(r.valid&&rate_error<.15)++stable[event];else stable[event]=0;
      if(stable[event]>=3&&!std::isfinite(metrics.recovery[event]))metrics.recovery[event]=t-changes[event]-2./30.;}
    bool transition=false;for(double event:changes)transition|=t+1e-9>=event&&t<event+.5;
    if(!transition)continue;
    ++metrics.frames;metrics.valid+=r.valid;metrics.adaptive_frames+=r.adaptive_window_active;
    metrics.omega_squared+=rate_error*rate_error;
    for(int s=0;s<3;++s){const auto truth=observe(c,center,manoeuvre_phase(t+h),s);
      for(int k=0;k<4;++k)if(r.slots[s].future_geometry_valid){const auto d=r.slots[s].future_points[k]-truth.measurement_points[k];
        ++metrics.geometric_corners;metrics.corner_squared+=d.dot(d);
        if(r.slots[s].future_valid){++metrics.valid_corners;metrics.valid_corner_squared+=d.dot(d);}}}
  }return metrics;
}
void complete_transition_regression(){
  for(double noise:{0.,.1}){const auto fixed=score_transitions(false,noise),adaptive=score_transitions(true,noise);
    const auto print=[&](const char* label,const TransitionMetrics& m){
      std::cout<<"transition "<<label<<" noise="<<noise<<" frames="<<m.frames<<" valid="<<m.valid
        <<" corner_coverage="<<m.valid_corners<<'/'<<m.frames*12<<" all_omega_rmse="<<std::sqrt(m.omega_squared/m.frames)
        <<" all_geometric_corner_rmse="<<std::sqrt(m.corner_squared/m.geometric_corners)
        <<" valid_corner_rmse="<<std::sqrt(m.valid_corner_squared/m.valid_corners)<<" recovery_s=";
      for(double delay:m.recovery)std::cout<<delay<<',';std::cout<<" adaptive_frames="<<m.adaptive_frames<<'\n';};
    print("fixed",fixed);print("adaptive",adaptive);
    require(fixed.frames==adaptive.frames&&adaptive.frames>=75,"transition denominator changed");
    require(adaptive.adaptive_frames>0,"coherent manoeuvre innovations never shortened window");
    require(adaptive.valid>=fixed.valid&&adaptive.valid_corners>=fixed.valid_corners,"accuracy improved by reducing coverage");
    require(adaptive.geometric_corners==fixed.geometric_corners,"geometric coverage changed between baselines");
    require(adaptive.omega_squared<fixed.omega_squared&&adaptive.corner_squared<fixed.corner_squared,"causal adaptation failed full-transition improvement");
    for(size_t k=0;k<fixed.recovery.size();++k){require(std::isfinite(adaptive.recovery[k]),"transition failed to recover");
      require(adaptive.recovery[k]<=fixed.recovery[k]+1./30.,"adaptation delayed transition recovery");}
  }
}
int main(){configuration_contracts();staggered_rotation_and_coast();stationary_reverse_and_multi_observation();translation_and_unknown_identity();assignment_ambiguity_contracts();initial_pose_score_contract();hypothesis_expiration_contract();noisy_degenerate_and_causality();
  measurement_boundary_contracts();constrained_projection_and_uncertainty();complete_transition_regression();
  std::cout<<"rigid estimator synthetic contracts passed; no real-video calibration or accuracy claim\n";}
