#include "rigid_estimator.hpp"
#include "covariance_fit.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace outpost {
namespace {
constexpr double tau=2*CV_PI,step=tau/3;
double wrap(double a){return std::remainder(a,tau);}
bool finite(const cv::Vec3d& x){return std::isfinite(x[0])&&std::isfinite(x[1])&&std::isfinite(x[2]);}
cv::Matx33d rotation(double phase,double pitch){
  const cv::Vec3d radial(std::cos(phase),std::sin(phase),0),tangent(-std::sin(phase),std::cos(phase),0);
  const auto up=std::sin(pitch)*radial+cv::Vec3d(0,0,std::cos(pitch));
  const auto normal=tangent.cross(up);
  return {tangent[0],up[0],normal[0],tangent[1],up[1],normal[1],tangent[2],up[2],normal[2]};
}
cv::Matx33d mat33(const cv::Mat& m){cv::Mat d;m.convertTo(d,CV_64F);return cv::Matx33d(d);}
bool quad_valid(const std::array<cv::Point2f,4>& p,double minimum){
  for(auto v:p)if(!std::isfinite(v.x)||!std::isfinite(v.y))return false;
  const std::vector<cv::Point2f> q(p.begin(),p.end());
  return cv::isContourConvex(q)&&std::abs(cv::contourArea(q))>=minimum;
}
using Cov4=cv::Matx44d;
using Jacobian=cv::Matx<double,8,4>;
double center_std(const Cov4& p){return std::sqrt(std::max(0.,p(0,0)+p(1,1)+p(2,2)));}
cv::Vec4d state(const cv::Vec3d& p,double a){return {p[0],p[1],p[2],a};}
Cov4 outer(const cv::Vec4d& v){return v*v.t();}
}
struct CalibratedRigidEstimator::Impl {
  struct Pose {int detection;cv::Vec3d armor;double phase,rms,orientation,information_ratio;Cov4 covariance;};
  struct Sample {double time;cv::Vec3d center;double phase,weight;Cov4 covariance;};
  struct Match {int pose=-1,slot=-1;double cost=1e100;};
  struct Branch {
    std::string fit_method="none",fit_selection_reason="not_fitted";
    double scalar_fit_score=NAN,gls_fit_score=NAN;
    cv::Vec3d center{},velocity{};double phase=0,omega=0,anchor=0,last_observation=0,score=0;
    double center_rms=0,phase_rms=0;int seen=0;bool motion=false;
    bool ownership_ambiguous=false,assignment_ambiguous_now=false;int recovery_hits=0;
    Cov4 position_covariance{},velocity_covariance{},cross_covariance{};
    bool covariance_valid=false;int change_hits=0;double adaptive_until=-1;
    int fitted_samples=0;double fitted_span=0;bool shortened=false;
    cv::Vec4d previous_innovation{};
    std::deque<Sample> samples;std::array<Match,3> matches;
  };
  RigidEstimatorConfig c;cv::Matx33d basis;std::vector<cv::Point3d> object;
  std::vector<Branch> branches;double last_time=-1;bool relative=false,resolved=false;
  std::uint64_t generation=0;
  explicit Impl(const RigidEstimatorConfig& config):c(config){
    auto positive=[](double x){return std::isfinite(x)&&x>0;};
    if(c.camera_matrix.rows!=3||c.camera_matrix.cols!=3||c.camera_matrix.channels()!=1||
       c.camera_from_target_rotation.rows!=3||c.camera_from_target_rotation.cols!=3||
       c.camera_from_target_rotation.channels()!=1||c.distortion.empty()||c.distortion.channels()!=1||
       (c.distortion.rows!=1&&c.distortion.cols!=1))throw std::runtime_error("rigid estimator requires explicit K, D and camera_from_target_rotation");
    c.camera_matrix.convertTo(c.camera_matrix,CV_64F);c.camera_matrix=c.camera_matrix.clone();
    c.distortion.convertTo(c.distortion,CV_64F);c.distortion=c.distortion.clone();
    if(!cv::checkRange(c.camera_matrix)||!cv::checkRange(c.distortion)||!cv::checkRange(c.camera_from_target_rotation))
      throw std::runtime_error("nonfinite rigid calibration");
    const auto k=mat33(c.camera_matrix);basis=mat33(c.camera_from_target_rotation);
    if(!positive(k(0,0))||!positive(k(1,1))||std::abs(k(2,2)-1)>1e-9||
       std::abs(k(1,0))+std::abs(k(2,0))+std::abs(k(2,1))+std::abs(k(0,1))>1e-9||
       cv::norm(cv::Mat(basis*basis.t()-cv::Matx33d::eye()))>1e-6||std::abs(cv::determinant(basis)-1)>1e-6)
      throw std::runtime_error("invalid rigid camera matrix or axis rotation");
    const auto n=c.distortion.total();if(n!=4&&n!=5&&n!=8&&n!=12&&n!=14)throw std::runtime_error("unsupported rigid distortion vector");
    for(double x:{c.lightbar_length_m,c.lightbar_center_distance_m,c.radius_m,c.history_seconds,
        c.min_history_seconds,c.coast_seconds,c.max_reprojection_error_px,c.max_orientation_error_rad,
        c.max_center_error_m,c.max_phase_error_rad,c.max_angular_speed_rad_s,c.max_center_speed_m_s,
        c.min_quad_area_px,c.identity_score_margin,c.pose_ambiguity_margin,c.max_fit_center_rms_m,c.max_fit_phase_rms_rad,
        c.pixel_noise_std_px,c.min_geometry_information_ratio,c.max_center_std_m,c.max_phase_std_rad,
        c.translation_acceleration_std_m_s2,c.angular_acceleration_std_rad_s2,c.max_future_corner_std_px,
        c.motion_change_threshold_rad,c.motion_change_threshold_m,c.association_nis_scale,c.robust_fit_gate_std})
      if(!positive(x))throw std::runtime_error("invalid rigid estimator parameter");
    if(!std::isfinite(c.pitch_rad)||std::abs(c.pitch_rad)>=CV_PI/2||c.min_samples<3||
       c.history_seconds<c.min_history_seconds||c.max_phase_error_rad>=CV_PI/3||c.max_orientation_error_rad>=CV_PI/2||
       c.pose_refinement_iterations<1||c.pose_refinement_iterations>100||c.min_geometry_information_ratio>=1||
       c.max_phase_std_rad>=CV_PI||c.motion_change_confirmations<2||c.motion_change_confirmations>100)
      throw std::runtime_error("invalid rigid estimator geometry/history/gate");
    for(double h:c.slot_heights_m)if(!std::isfinite(h))throw std::runtime_error("three explicit slot heights required");
    relative=std::abs(c.slot_heights_m[0]-c.slot_heights_m[1])<1e-9&&
             std::abs(c.slot_heights_m[0]-c.slot_heights_m[2])<1e-9;
    const double w=c.lightbar_center_distance_m/2,h=c.lightbar_length_m/2;
    object={{-w,h,0},{w,h,0},{w,-h,0},{-w,-h,0}};
  }
  cv::Vec3d center_from(const Pose& p,int slot)const{
    return p.armor-c.radius_m*cv::Vec3d(std::cos(p.phase),std::sin(p.phase),0)-cv::Vec3d(0,0,c.slot_heights_m[slot]);
  }
  // The free PnP orientation is an initializer sanity gate only. Scores and
  // weights use the same constrained geometry that is subsequently projected.
  double pose_score(const Pose& p)const{return p.rms/c.max_reprojection_error_px;}
  bool distinct_pose(const Pose& a,const Pose& b)const{
    return std::abs(wrap(a.phase-b.phase))/c.max_phase_error_rad+
           cv::norm(a.armor-b.armor)/c.max_center_error_m>c.pose_ambiguity_margin;
  }
  bool project_armor(const cv::Vec4d& x,std::vector<cv::Point2d>& pixels,bool front=true)const{
    const auto r=basis*rotation(x[3],c.pitch_rad);
    const auto t=basis*cv::Vec3d(x[0],x[1],x[2]);
    if(!finite(t)||(front&&cv::Vec3d(r(0,2),r(1,2),r(2,2)).dot(t)>=0))return false;
    for(auto q:object)if((r*cv::Vec3d(q.x,q.y,q.z)+t)[2]<=1e-6)return false;
    cv::Mat rv;cv::Rodrigues(cv::Mat(r),rv);
    cv::projectPoints(object,rv,t,c.camera_matrix,c.distortion,pixels);
    for(auto p:pixels)if(!std::isfinite(p.x)||!std::isfinite(p.y))return false;
    return true;
  }
  bool armor_jacobian(const cv::Vec4d& x,Jacobian& j)const{
    for(int k=0;k<4;++k){const double epsilon=k==3?1e-5:1e-5*c.radius_m;
      auto a=x,b=x;a[k]+=epsilon;b[k]-=epsilon;std::vector<cv::Point2d> pa,pb;
      if(!project_armor(a,pa)||!project_armor(b,pb))return false;
      for(int i=0;i<4;++i){j(2*i,k)=(pa[i].x-pb[i].x)/(2*epsilon);j(2*i+1,k)=(pa[i].y-pb[i].y)/(2*epsilon);}}
    return true;
  }
  bool refine_pose(Pose& p,const std::array<cv::Point2f,4>& measured)const{
    auto x=state(p.armor,p.phase);double damping=1e-3;
    const auto residual=[&](const cv::Vec4d& q,cv::Vec<double,8>& error){
      std::vector<cv::Point2d> projected;if(!project_armor(q,projected))return false;
      for(int i=0;i<4;++i){error[2*i]=projected[i].x-measured[i].x;error[2*i+1]=projected[i].y-measured[i].y;}return true;};
    cv::Vec<double,8> error;if(!residual(x,error))return false;
    for(int iteration=0;iteration<c.pose_refinement_iterations;++iteration){
      Jacobian j;if(!armor_jacobian(x,j))return false;
      for(int i=0;i<8;++i)for(int k=0;k<3;++k)j(i,k)*=c.radius_m;
      auto normal=j.t()*j;const auto gradient=j.t()*error;
      for(int k=0;k<4;++k)normal(k,k)+=damping*std::max(1.,normal(k,k));
      cv::Vec4d delta;if(!cv::solve(normal,-gradient,delta,cv::DECOMP_CHOLESKY))return false;
      auto next=x;for(int k=0;k<4;++k)next[k]+=delta[k]*(k==3?1.:c.radius_m);
      cv::Vec<double,8> next_error;
      if(residual(next,next_error)&&next_error.dot(next_error)<error.dot(error)){
        x=next;error=next_error;damping=std::max(1e-9,damping*.3);if(cv::norm(delta)<1e-8)break;
      }else damping=std::min(1e12,damping*10);
    }
    p.rms=std::sqrt(error.dot(error)/4);
    if(!std::isfinite(p.rms)||p.rms>c.max_reprojection_error_px)return false;
    Jacobian j;if(!armor_jacobian(x,j))return false;
    auto scaled=j;for(int i=0;i<8;++i)for(int k=0;k<3;++k)scaled(i,k)*=c.radius_m;
    cv::Mat eigenvalues;cv::eigen(cv::Mat(scaled.t()*scaled),eigenvalues);
    p.information_ratio=eigenvalues.at<double>(3)/std::max(1e-30,eigenvalues.at<double>(0));
    if(!std::isfinite(p.information_ratio)||p.information_ratio<c.min_geometry_information_ratio)return false;
    const double variance=std::max(c.pixel_noise_std_px*c.pixel_noise_std_px,error.dot(error)/4);
    p.covariance=(j.t()*j).inv(cv::DECOMP_SVD)*variance;
    p.armor={x[0],x[1],x[2]};p.phase=wrap(x[3]);return cv::checkRange(cv::Mat(p.covariance));
  }
  Cov4 center_covariance(const Pose& p)const{
    auto transform=Cov4::eye();transform(0,3)=c.radius_m*std::sin(p.phase);transform(1,3)=-c.radius_m*std::cos(p.phase);
    return transform*p.covariance*transform.t();
  }
  std::vector<Pose> poses(const std::vector<Detection>& detections,std::vector<RigidMeasurementDiagnostic>& diagnostics)const{
    std::vector<Pose> out;
    for(size_t i=0;i<detections.size();++i){
      diagnostics.push_back({int(i)});auto& diagnostic=diagnostics.back();const auto& d=detections[i];
      if(d.kind==ObservationKind::NeuralKeypoints){diagnostic.reason="centerline_measurement_required";continue;}
      // Validate the measurement contract here too. Other backends/replay
      // callers may bypass the traditional pairer's boundary rejection.
      if(std::any_of(d.endpoint_evidence.begin(),d.endpoint_evidence.end(),[](auto evidence){
          return evidence==EndpointEvidence::BoundaryUnsupported;
        })){diagnostic.reason="boundary_endpoint_support_missing";continue;}
      if(!quad_valid(d.measurement_points,c.min_quad_area_px)){diagnostic.reason="degenerate_measurement";continue;}
      std::vector<cv::Mat> rvecs,tvecs;
      try{cv::solvePnPGeneric(object,std::vector<cv::Point2f>(d.measurement_points.begin(),d.measurement_points.end()),
          c.camera_matrix,c.distortion,rvecs,tvecs,false,cv::SOLVEPNP_IPPE);}
      catch(const cv::Exception&){diagnostic.reason="pnp_failed";continue;}
      for(size_t j=0;j<rvecs.size();++j){
        cv::Mat rmat;cv::Rodrigues(rvecs[j],rmat);const auto rc=mat33(rmat);
        cv::Mat td;tvecs[j].convertTo(td,CV_64F);const cv::Vec3d tc(td.at<double>(0),td.at<double>(1),td.at<double>(2));
        if(!finite(tc))continue;
        bool positive_depth=true;for(auto q:object)if((rc*cv::Vec3d(q.x,q.y,q.z)+tc)[2]<=1e-6)positive_depth=false;
        // The measured left/right correspondence describes the outward front
        // face. A rear-facing pose cannot explain an accepted visible lamp pair.
        if(!positive_depth||cv::Vec3d(rc(0,2),rc(1,2),rc(2,2)).dot(tc)>=0)continue;
        const auto rt=basis.t()*rc;const double phase=std::atan2(-rt(0,0),rt(1,0));
        const auto error=rotation(phase,c.pitch_rad).t()*rt;
        const double orientation=std::acos(std::clamp((cv::trace(cv::Mat(error))[0]-1)*.5,-1.,1.));
        if(orientation>c.max_orientation_error_rad)continue;
        Pose p{int(i),basis.t()*tc,phase,0,orientation,0,{}};
        if(!refine_pose(p,d.measurement_points)){diagnostic.reason="constrained_pose_or_information_rejected";continue;}
        // Multiple IPPE starts can converge to the same constrained minimum.
        // They are not separate identity/pose evidence.
        bool duplicate=false;for(auto& old:out)if(old.detection==int(i)&&
          std::abs(wrap(old.phase-p.phase))<1e-6&&cv::norm(old.armor-p.armor)<1e-6){
          if(p.rms<old.rms)old=p;
          duplicate=true;break;}
        if(!duplicate)out.push_back(p);
        diagnostic.reason="pose_not_associated";diagnostic.reprojection_error_px=p.rms;diagnostic.orientation_error_rad=orientation;
        diagnostic.geometry_information_ratio=p.information_ratio;const auto covariance=center_covariance(p);
        diagnostic.center_std_m=center_std(covariance);diagnostic.phase_std_rad=std::sqrt(std::max(0.,covariance(3,3)));
      }
    }return out;
  }
  void fit(Branch& b){
    b.fit_method="scalar_robust";b.fit_selection_reason=c.covariance_estimation_enabled?"insufficient_samples":"covariance_disabled";
    b.scalar_fit_score=NAN;b.gls_fit_score=NAN;
    const size_t n=b.samples.size();b.anchor=b.samples.back().time;
    std::vector<double> weights;for(const auto& s:b.samples)weights.push_back(s.weight);
    cv::Vec4d intercept{},slope{};
    for(int iteration=0;iteration<4;++iteration){
      double sw=0,st=0,stt=0;cv::Vec4d sy{},sty{};
      for(size_t i=0;i<n;++i){const auto& s=b.samples[i];const double t=s.time-b.anchor,w=weights[i];
        const cv::Vec4d y(s.center[0],s.center[1],s.center[2],s.phase);sw+=w;st+=w*t;stt+=w*t*t;sy+=w*y;sty+=w*t*y;}
      const double determinant=sw*stt-st*st;
      slope=determinant>1e-12?(sw*sty-st*sy)/determinant:cv::Vec4d();
      intercept=(sy-st*slope)/sw;
      if(iteration==3)break; // Covariance below must use the weights actually fitted.
      for(size_t i=0;i<n;++i){const auto& s=b.samples[i];const auto residual=cv::Vec4d(s.center[0],s.center[1],s.center[2],s.phase)-intercept-(s.time-b.anchor)*slope;
        const double normalized=std::sqrt((residual[0]*residual[0]+residual[1]*residual[1]+residual[2]*residual[2])/
           (c.max_fit_center_rms_m*c.max_fit_center_rms_m)+residual[3]*residual[3]/(c.max_fit_phase_rms_rad*c.max_fit_phase_rms_rad));
        weights[i]=s.weight/std::max(1.,normalized);
      }
    }
    b.center={intercept[0],intercept[1],intercept[2]};b.velocity={slope[0],slope[1],slope[2]};b.phase=intercept[3];b.omega=slope[3];
    b.center_rms=0;b.phase_rms=0;
    for(const auto& s:b.samples){const auto dc=s.center-b.center-(s.time-b.anchor)*b.velocity;
      const double da=s.phase-b.phase-(s.time-b.anchor)*b.omega;b.center_rms+=dc.dot(dc);b.phase_rms+=da*da;}
    b.center_rms=std::sqrt(b.center_rms/n);b.phase_rms=std::sqrt(b.phase_rms/n);
    // Linearised independent-observation WLS covariance, with the empirical
    // residual scatter added for unmodelled within-window motion. This is a
    // modelling uncertainty, not a claimed calibrated confidence probability.
    double sw=0,st=0,stt=0;Cov4 scatter{};
    for(size_t i=0;i<n;++i){const auto& s=b.samples[i];const double t=s.time-b.anchor,w=weights[i];
      sw+=w;st+=w*t;stt+=w*t*t;scatter+=outer(state(s.center,s.phase)-intercept-t*slope);}
    scatter*=1./std::max(1.,double(n)-2);const double determinant=sw*stt-st*st;
    b.position_covariance=Cov4();b.velocity_covariance=Cov4();b.cross_covariance=Cov4();
    b.covariance_valid=n>=2&&determinant>1e-12;
    if(b.covariance_valid)for(size_t i=0;i<n;++i){const double t=b.samples[i].time-b.anchor,w=weights[i];
      const double a=w*(stt-st*t)/determinant,v=w*(sw*t-st)/determinant;
      const auto covariance=b.samples[i].covariance+scatter;
      b.position_covariance+=a*a*covariance;b.velocity_covariance+=v*v*covariance;b.cross_covariance+=a*v*covariance;}
    b.scalar_fit_score=std::pow(b.center_rms/c.max_fit_center_rms_m,2)+std::pow(b.phase_rms/c.max_fit_phase_rms_rad,2);
    if(c.covariance_estimation_enabled&&n>=2){
      std::vector<CovarianceSample> samples;samples.reserve(n);
      for(const auto& s:b.samples)samples.push_back({s.time,state(s.center,s.phase),s.covariance});
      try{const auto fit=covariance_line_fit(samples,b.anchor,c.robust_fit_gate_std);
        double center_squared=0,phase_squared=0;
        for(const auto& s:b.samples){const auto residual=state(s.center,s.phase)-fit.position-(s.time-b.anchor)*fit.velocity;
          for(int k=0;k<3;++k)center_squared+=residual[k]*residual[k];
          phase_squared+=residual[3]*residual[3];}
        const double score=center_squared/(n*c.max_fit_center_rms_m*c.max_fit_center_rms_m)+
          phase_squared/(n*c.max_fit_phase_rms_rad*c.max_fit_phase_rms_rad);
        const double baseline=std::pow(b.center_rms/c.max_fit_center_rms_m,2)+std::pow(b.phase_rms/c.max_fit_phase_rms_rad,2);
        b.gls_fit_score=score;b.fit_selection_reason="gls_residual_worse";
        // Correlated pose covariance is only a local linear approximation.
        // Retain the robust scalar fit if GLS worsens the independently checked
        // centre/phase residuals, especially across non-linear manoeuvres.
        if(score<=baseline+1e-10){
        b.fit_method="covariance_gls";b.fit_selection_reason="gls_residual_accepted";
        b.center={fit.position[0],fit.position[1],fit.position[2]};b.phase=fit.position[3];
        b.velocity={fit.velocity[0],fit.velocity[1],fit.velocity[2]};b.omega=fit.velocity[3];
        b.position_covariance=fit.position_covariance;b.velocity_covariance=fit.velocity_covariance;
        b.cross_covariance=fit.cross_covariance;b.covariance_valid=true;
        b.center_rms=0;b.phase_rms=0;
        for(const auto& s:b.samples){const auto dc=s.center-b.center-(s.time-b.anchor)*b.velocity;
          const auto da=s.phase-b.phase-(s.time-b.anchor)*b.omega;b.center_rms+=dc.dot(dc);b.phase_rms+=da*da;}
        b.center_rms=std::sqrt(b.center_rms/n);b.phase_rms=std::sqrt(b.phase_rms/n);
        }
      }catch(const std::runtime_error&){b.fit_selection_reason="gls_numerical_failure";}
    }
    b.motion=int(n)>=c.min_samples&&b.samples.back().time-b.samples.front().time+1e-9>=c.min_history_seconds&&
      b.center_rms<=c.max_fit_center_rms_m&&b.phase_rms<=c.max_fit_phase_rms_rad&&
      std::abs(b.omega)<=c.max_angular_speed_rad_s&&cv::norm(b.velocity)<=c.max_center_speed_m_s;
  }
  Cov4 extrapolated_covariance(const Branch& b,double dt)const{
    auto p=b.position_covariance+dt*(b.cross_covariance+b.cross_covariance.t())+dt*dt*b.velocity_covariance;
    const double acceleration_factor=.25*dt*dt*dt*dt;
    for(int i=0;i<4;++i){const double a=i==3?c.angular_acceleration_std_rad_s2:c.translation_acceleration_std_m_s2;p(i,i)+=acceleration_factor*a*a;}
    return p;
  }
  void consume(Branch& b,double time,const std::vector<Pose>& poses,size_t detection_count,double horizon){
    b.assignment_ambiguous_now=false;
    const double dt=time-b.anchor;const auto predicted_center=b.center+dt*b.velocity;
    const double predicted_phase=b.phase+dt*b.omega;
    std::array<std::vector<Match>,3> options;
    for(int slot=0;slot<3;++slot)for(size_t d=0;d<detection_count;++d){
      Match best,runner;for(size_t i=0;i<poses.size();++i){const auto& p=poses[i];if(p.detection!=int(d))continue;
        const double ce=cv::norm(center_from(p,slot)-predicted_center),pe=std::abs(wrap(p.phase-predicted_phase-slot*step));
        if(ce>c.max_center_error_m||pe>c.max_phase_error_rad)continue;
        double innovation_cost=std::pow(ce/c.max_center_error_m,2)+std::pow(pe/c.max_phase_error_rad,2);
        if(c.covariance_estimation_enabled&&b.motion&&b.covariance_valid){
          const auto residual=state(center_from(p,slot)-predicted_center,wrap(p.phase-predicted_phase-slot*step));
          const auto covariance=extrapolated_covariance(b,dt)+center_covariance(p);
          cv::Vec4d normalized;
          if(!cv::solve(covariance,residual,normalized,cv::DECOMP_CHOLESKY))continue;
          const double nis=residual.dot(normalized);
          if(!std::isfinite(nis)||nis<0)continue;
          // A short-window constant-velocity model is knowingly imperfect at
          // starts/stops. NIS ranks candidates but is not a second hard gate:
          // rejecting coherent manoeuvre evidence here prevents adaptation.
          // Fixed physical sanity bounds above still reject implausible jumps.
          innovation_cost=nis/(c.association_nis_scale+nis);
        }
        const double cost=innovation_cost+std::pow(p.rms/c.max_reprojection_error_px,2);
        if(cost<best.cost){runner=best;best={int(i),slot,cost};}else if(cost<runner.cost)runner={int(i),slot,cost};
      }
      if(best.pose>=0){
        if(runner.pose>=0&&runner.cost-best.cost<c.pose_ambiguity_margin&&
           distinct_pose(poses[best.pose],poses[runner.pose]))continue;
        options[slot].push_back(best);
      }
    }
    std::array<Match,3> selected,current;double best_cost=1e100,runner_cost=1e100;
    std::vector<bool> used(detection_count,false);
    std::function<void(int,double)> search=[&](int slot,double cost){
      if(cost>runner_cost)return;
      if(slot==3){
        if(cost<best_cost){runner_cost=best_cost;best_cost=cost;selected=current;}
        else if(cost<runner_cost)runner_cost=cost;
        return;
      }
      current[slot]=Match{};search(slot+1,cost+4);
      for(const auto& m:options[slot]){const int d=poses[m.pose].detection;if(used[d])continue;
        used[d]=true;current[slot]=m;search(slot+1,cost+m.cost);used[d]=false;}
    };search(0,0);
    // Ownership ambiguity affects the whole joint assignment. Keeping only the
    // first equal-cost matching would make identity depend on candidate order.
    // Neither an ambiguous measurement nor an empty frame clears this latch.
    if(runner_cost-best_cost<c.pose_ambiguity_margin){
      b.ownership_ambiguous=true;b.assignment_ambiguous_now=true;b.recovery_hits=0;b.matches={};
      if(!poses.empty())b.score+=best_cost;
      return;
    }
    b.matches=selected;
    cv::Vec3d center{};double phase=0,weight=0;int accepted=0;Cov4 covariance{},information{};cv::Vec4d information_value{};
    for(int slot=0;slot<3;++slot){const auto m=selected[slot];if(m.pose<0)continue;const auto& p=poses[m.pose];
      const double w=1/(c.pixel_noise_std_px*c.pixel_noise_std_px+p.rms*p.rms);center+=w*center_from(p,slot);
      const double a=predicted_phase+wrap(p.phase-slot*step-predicted_phase);
      phase+=w*a;weight+=w;covariance+=w*w*center_covariance(p);
      if(c.covariance_estimation_enabled){const auto inv=center_covariance(p).inv(cv::DECOMP_CHOLESKY);
        information+=inv;information_value+=inv*state(center_from(p,slot),a);}
      ++accepted;b.seen|=1<<slot;}
    // Common missed-slot constants cancel between hypotheses. This score is
    // solely a relative association cost, not a calibrated probability.
    if(!poses.empty())b.score+=best_cost;
    if(!accepted){b.recovery_hits=0;b.change_hits=0;
      // With no new evidence the short manoeuvre fit is no longer refreshed.
      // Reuse the retained full causal window for a less uncertain coast.
      if(b.shortened){fit(b);b.fitted_samples=int(b.samples.size());
        b.fitted_span=b.samples.back().time-b.samples.front().time;b.shortened=false;}
      return;}
    if(b.ownership_ambiguous&&++b.recovery_hits>=c.min_samples){b.ownership_ambiguous=false;b.recovery_hits=0;}
    center/=weight;phase/=weight;covariance*=1/(weight*weight);b.last_observation=time;
    if(c.covariance_estimation_enabled){covariance=information.inv(cv::DECOMP_CHOLESKY);
      const auto fused=covariance*information_value;center={fused[0],fused[1],fused[2]};phase=fused[3];}
    const auto innovation=state(center-predicted_center,phase-predicted_phase);
    if(c.adaptive_window_enabled&&b.motion){
      const bool phase_change=std::abs(innovation[3])>c.motion_change_threshold_rad;
      const cv::Vec3d dc(innovation[0],innovation[1],innovation[2]);
      const cv::Vec3d previous(b.previous_innovation[0],b.previous_innovation[1],b.previous_innovation[2]);
      const bool center_change=cv::norm(dc)>c.motion_change_threshold_m;
      const bool coherent=(phase_change&&innovation[3]*b.previous_innovation[3]>0)||(center_change&&dc.dot(previous)>0);
      b.change_hits=(phase_change||center_change)?(coherent?b.change_hits+1:1):0;
      if(b.change_hits>=c.motion_change_confirmations){b.adaptive_until=time+c.history_seconds;b.change_hits=0;}
    }else b.change_hits=0;
    b.previous_innovation=innovation;
    b.samples.push_back({time,center,phase,weight,covariance});
    while(b.samples.size()>1&&time-b.samples.front().time>c.history_seconds+1e-9)b.samples.pop_front();
    // Shorten only after repeated coherent innovations; retain both the minimum
    // time support and sample count. No invented speed or retrospective samples.
    fit(b);
    b.fitted_samples=int(b.samples.size());b.fitted_span=b.samples.back().time-b.samples.front().time;b.shortened=false;
    if(c.adaptive_window_enabled&&time<=b.adaptive_until){
      // Keep the full causal history. A shorter fit must pass the same state
      // and requested-horizon uncertainty gates before replacing the full fit.
      // On the next observation the full window remains available again.
      const auto full_history=b.samples;auto trial=b;
      while(int(trial.samples.size())>c.min_samples&&time-trial.samples[1].time+1e-9>=c.min_history_seconds){
        trial.samples.pop_front();fit(trial);
        if(!trial.motion||!trial.covariance_valid)continue;
        const auto covariance=extrapolated_covariance(trial,0);
        bool reliable=center_std(covariance)<=c.max_center_std_m&&
          std::sqrt(std::max(0.,covariance(3,3)))<=c.max_phase_std_rad;
        const auto future=state(trial.center+horizon*trial.velocity,trial.phase+horizon*trial.omega);
        for(int s=0;s<3&&reliable;++s){RigidProjectedSlot projected;
          reliable=corner_uncertainty(future,s,extrapolated_covariance(trial,horizon),projected)&&
            projected.max_future_corner_std_px<=c.max_future_corner_std_px;}
        if(reliable){b=trial;b.fitted_samples=int(trial.samples.size());
          b.fitted_span=trial.samples.back().time-trial.samples.front().time;b.shortened=true;}
      }
      b.samples=full_history;
    }
  }
  bool project(const cv::Vec3d& center,double phase,int slot,std::array<cv::Point2f,4>& out)const{
    std::vector<cv::Point2d> p;if(!project_center(state(center,phase),slot,p))return false;
    for(int k=0;k<4;++k)out[k]=p[k];
    return quad_valid(out,.01);
  }
  bool project_center(const cv::Vec4d& x,int slot,std::vector<cv::Point2d>& pixels)const{
    const double a=x[3]+slot*step;
    const auto armor=cv::Vec3d(x[0],x[1],x[2])+c.radius_m*cv::Vec3d(std::cos(a),std::sin(a),0)+cv::Vec3d(0,0,c.slot_heights_m[slot]);
    return project_armor(state(armor,a),pixels,false);
  }
  bool corner_uncertainty(const cv::Vec4d& x,int slot,const Cov4& covariance,RigidProjectedSlot& out)const{
    Jacobian j;
    for(int k=0;k<4;++k){const double epsilon=k==3?1e-5:1e-5*c.radius_m;
      auto a=x,b=x;a[k]+=epsilon;b[k]-=epsilon;std::vector<cv::Point2d> pa,pb;
      if(!project_center(a,slot,pa)||!project_center(b,slot,pb))return false;
      for(int i=0;i<4;++i){j(2*i,k)=(pa[i].x-pb[i].x)/(2*epsilon);j(2*i+1,k)=(pa[i].y-pb[i].y)/(2*epsilon);}}
    const auto p=j*covariance*j.t();double maximum=0;
    for(int i=0;i<4;++i){const double a=p(2*i,2*i),d=p(2*i+1,2*i+1),b=p(2*i,2*i+1);
      const double eigenvalue=.5*(a+d+std::sqrt(std::max(0.,(a-d)*(a-d)+4*b*b)));
      if(!std::isfinite(eigenvalue)||eigenvalue<0)return false;
      out.future_corner_std_px[i]=std::sqrt(eigenvalue);maximum=std::max(maximum,out.future_corner_std_px[i]);}
    out.max_future_corner_std_px=maximum;return true;
  }
  RigidEstimate update(double time,const std::vector<Detection>& detections,double horizon){
    if(!std::isfinite(time)||time<0||(last_time>=0&&time<=last_time)||!std::isfinite(horizon)||horizon<0)
      throw std::runtime_error("invalid rigid timestamp or horizon");
    last_time=time;RigidEstimate out;out.time=time;out.relative_identity_only=relative;out.generation=generation;
    for(int s=0;s<3;++s)out.slots[s].id=s;
    const auto measurements=poses(detections,out.diagnostics);
    if(!branches.empty()){
      // Expiration belongs to each hypothesis, not only to the selected one.
      // An old hypothesis must never consume a new observation and resurrect
      // its accumulated score, visited identities or motion fit.
      branches.erase(std::remove_if(branches.begin(),branches.end(),[&](const auto& b){
        return time-b.last_observation>c.coast_seconds;
      }),branches.end());
      // Pruning does not constitute positive identity evidence. An unresolved
      // singleton restarts all cyclic hypotheses instead of becoming "certain"
      // merely because its alternatives expired. Resolved singletons may coast.
      if(branches.empty()||(!resolved&&branches.size()<2)){
        branches.clear();resolved=false;
      }
    }
    if(branches.empty()){
      if(measurements.empty())return out;
      const auto seed=std::min_element(measurements.begin(),measurements.end(),[&](const auto& a,const auto& b){return pose_score(a)<pose_score(b);});
      for(const auto& p:measurements)if(&p!=&*seed&&p.detection==seed->detection&&
        pose_score(p)-pose_score(*seed)<c.pose_ambiguity_margin&&distinct_pose(p,*seed)){
        out.diagnostics[seed->detection].reason="ambiguous_initial_pose";return out;
      }
      for(int s=0;s<(relative?1:3);++s){Branch b;b.center=center_from(*seed,s);b.phase=seed->phase-s*step;
        b.anchor=time;b.last_observation=time;branches.push_back(b);}
      resolved=relative;out.generation=++generation;
    }
    for(auto& b:branches)consume(b,time,measurements,detections.size(),horizon);
    std::stable_sort(branches.begin(),branches.end(),[](const auto& a,const auto& b){return a.score<b.score;});
    if(!resolved&&branches.size()>1&&branches[1].score-branches[0].score>=c.identity_score_margin&&
       (branches[0].seen&(branches[0].seen-1))!=0&&branches[0].motion&&!branches[0].ownership_ambiguous){branches.resize(1);resolved=true;}
    const auto& b=branches.front();const double dt=time-b.anchor;
    out.initialized=true;out.identity_resolved=resolved;out.motion_confirmed=b.motion&&!b.ownership_ambiguous;
    out.association_ambiguous=b.ownership_ambiguous;
    out.center=b.center+dt*b.velocity;out.velocity=b.velocity;out.phase=wrap(b.phase+dt*b.omega);out.omega=b.omega;
    out.age_seconds=time-b.last_observation;out.fit_center_rms_m=b.center_rms;out.fit_phase_rms_rad=b.phase_rms;
    out.sample_count=b.fitted_samples;out.uncertainty_available=b.covariance_valid;
    out.active_history_seconds=b.fitted_span;
    out.adaptive_window_active=b.shortened;
    out.fit_method=b.fit_method;out.fit_selection_reason=b.fit_selection_reason;
    out.scalar_fit_score=b.scalar_fit_score;out.gls_fit_score=b.gls_fit_score;
    if(out.uncertainty_available){const auto p=extrapolated_covariance(b,dt);
      out.center_std_m=center_std(p);out.phase_std_rad=std::sqrt(std::max(0.,p(3,3)));
      out.omega_std_rad_s=std::sqrt(std::max(0.,b.velocity_covariance(3,3)));
      out.state_reliable=out.center_std_m<=c.max_center_std_m&&out.phase_std_rad<=c.max_phase_std_rad;}
    out.valid=resolved&&out.motion_confirmed&&out.state_reliable&&out.age_seconds<=c.coast_seconds;
    if(b.assignment_ambiguous_now)for(auto& d:out.diagnostics)
      if(d.reason=="pose_not_associated")d.reason="ambiguous_assignment";
    for(int s=0;s<3;++s){auto& slot=out.slots[s];const auto m=b.matches[s];
      if(m.pose>=0){const auto& p=measurements[m.pose];slot.observed=true;slot.detection_index=p.detection;
        auto& diagnostic=out.diagnostics[p.detection];diagnostic.accepted=true;diagnostic.slot=s;
        diagnostic.reason=!resolved?"accepted_identity_unresolved":
          b.ownership_ambiguous?"accepted_ownership_recovering":"accepted";diagnostic.reprojection_error_px=p.rms;
        diagnostic.orientation_error_rad=p.orientation;diagnostic.geometry_information_ratio=p.information_ratio;
        const auto covariance=center_covariance(p);diagnostic.center_std_m=center_std(covariance);
        diagnostic.phase_std_rad=std::sqrt(std::max(0.,covariance(3,3)));++out.accepted_measurements;}
      slot.current_valid=resolved&&project(out.center,out.phase,s,slot.current_points);
      const auto future=state(out.center+horizon*out.velocity,out.phase+horizon*out.omega);
      slot.future_geometry_valid=project(cv::Vec3d(future[0],future[1],future[2]),future[3],s,slot.future_points);
      // Diagnostic availability must not depend on drawing/identity permission.
      // A blocked forecast may still have a finite, useful uncertainty estimate.
      slot.future_uncertainty_available=slot.future_geometry_valid&&out.uncertainty_available&&
        corner_uncertainty(future,s,extrapolated_covariance(b,dt+horizon),slot);
      slot.prediction_reliable=out.valid&&out.age_seconds+horizon<=c.coast_seconds&&
        slot.future_uncertainty_available&&slot.max_future_corner_std_px<=c.max_future_corner_std_px;
      slot.future_valid=slot.future_geometry_valid&&slot.prediction_reliable;
      slot.forecast_block_reason=slot.future_valid?"none":
        !resolved?"identity_unresolved":out.association_ambiguous?"association_ambiguous":
        !out.motion_confirmed?"motion_unconfirmed":!out.state_reliable?"state_uncertain":
        out.age_seconds+horizon>c.coast_seconds?"horizon_expired":
        !slot.future_geometry_valid?"invalid_projection":
        !slot.future_uncertainty_available?"uncertainty_unavailable":"corner_uncertainty_exceeded";
    }
    out.measurement_used=out.accepted_measurements>0;
    out.state=!out.valid?RigidTrackingState::Initializing:
      (out.measurement_used?RigidTrackingState::Tracking:RigidTrackingState::Coasting);
    return out;
  }
};
CalibratedRigidEstimator::CalibratedRigidEstimator(const RigidEstimatorConfig& c):impl_(std::make_unique<Impl>(c)){}
CalibratedRigidEstimator::~CalibratedRigidEstimator()=default;
CalibratedRigidEstimator::CalibratedRigidEstimator(CalibratedRigidEstimator&&) noexcept=default;
CalibratedRigidEstimator& CalibratedRigidEstimator::operator=(CalibratedRigidEstimator&&) noexcept=default;
RigidEstimate CalibratedRigidEstimator::update(double t,const std::vector<Detection>& d,double h){return impl_->update(t,d,h);}
}
