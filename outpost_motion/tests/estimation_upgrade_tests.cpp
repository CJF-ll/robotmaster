#include "covariance_fit.hpp"
#include "rigid_estimator.hpp"
#include "tracker.hpp"
#include "visibility.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
using namespace outpost;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void covariance_contract(){
  std::vector<CovarianceSample> samples;
  for(int i=0;i<12;++i){const double t=i*.04;
    cv::Matx44d covariance=cv::Matx44d::eye()*.001;
    const bool weak=i>=8;if(weak)covariance(0,0)=100;
    covariance(1,2)=covariance(2,1)=.0002;
    samples.push_back({t,{1+2*t+(weak?5:0),2-t,3+.2*t,.3+.7*t},covariance});}
  const auto r=covariance_line_fit(samples,.44);
  check(std::abs(r.velocity[0]-2)<.01,"anisotropic low-quality coordinate biased GLS velocity");
  check(std::abs(r.velocity[1]+1)<1e-10&&std::abs(r.velocity[3]-.7)<1e-10,"cross-coordinate covariance changed exact motion");
  check(r.position_covariance(0,0)>r.position_covariance(1,1),"weak coordinate uncertainty lost");
  auto invalid=samples;invalid[0].covariance(0,0)=-1;bool rejected=false;
  try{covariance_line_fit(invalid,.44);}catch(...){rejected=true;}
  check(rejected,"negative covariance accepted");
}
Detection box(double center,double half_width){
  Detection d;d.center={float(center),60};
  d.measurement_points={{{float(center-half_width),50},{float(center+half_width),50},
                         {float(center+half_width),70},{float(center-half_width),70}}};
  d.corners=d.measurement_points;return d;
}
void shape_contract(){
  TrackerConfig cfg;cfg.shape_guard_enabled=true;Tracker guarded(cfg);cfg.shape_guard_enabled=false;Tracker baseline(cfg);
  int blocked=0;std::uint64_t id=0;
  for(int i=0;i<30;++i){const double width=i<10?40:i<20?40-2*(i-9):20;
    const auto d=box(120+2*i,width);const auto a=guarded.step(i/30.,{d},.1),b=baseline.step(i/30.,{d},.1);
    if(a.empty())continue;
    check(a.size()==1&&b.size()==1,"shape guard changed observation ownership");
    if(!id)id=a[0].id;check(a[0].id==id,"shape gate changed identity");
    check(cv::norm(a[0].future_center-b[0].future_center)<1e-9,"shape gate contaminated center prediction");
    if(i>=5&&i<10)check(a[0].future_center_valid&&a[0].future_shape_reliable&&a[0].future_valid,"pure translation rejected as shape change");
    if(i>=12&&i<20){check(a[0].future_center_valid&&!a[0].future_shape_reliable&&!a[0].future_valid,
      "rapidly narrowing shape presented as reliable full quad");++blocked;}
    if(i>=24)check(a[0].future_valid,"stable measured silhouette did not recover");
  }
  check(blocked==8,"shape test coverage incomplete");
  // Numerical boundary fixture only, not a physical operating horizon.
  // Float coordinates lose quad width far before a finite center overflows.
  TrackerConfig wide;wide.coast_seconds=1e9;wide.max_position_std=1e15;Tracker finite_center(wide);
  std::vector<TrackOutput> result;
  for(int i=0;i<20;++i)result=finite_center.step(i/30.,{box(120+2*i,10)},1e8);
  check(result.size()==1&&result[0].future_center_valid&&!result[0].future_valid&&
    (result[0].forecast.blocks&ForecastGeometry),"invalid future silhouette revoked finite center state");
  auto invalid=result[0];invalid.current.fill(invalid.current_center);
  const auto drawing=visibility(invalid,{},VisibilityConfig{});
  check(!drawing.current&&!drawing.future&&invalid.future_center_valid,"render boundary accepted collapsed quad or changed center");
}
RigidEstimatorConfig camera(bool staggered){
  RigidEstimatorConfig c;c.camera_matrix=(cv::Mat_<double>(3,3)<<1100,0,640,0,1100,360,0,0,1);
  c.distortion=(cv::Mat_<double>(1,5)<<-.06,.01,.001,-.001,0);
  c.camera_from_target_rotation=(cv::Mat_<double>(3,3)<<0,1,0,0,0,-1,-1,0,0);
  c.lightbar_length_m=.06;c.lightbar_center_distance_m=.135;c.radius_m=.27;c.pitch_rad=-.15;
  c.slot_heights_m=staggered?std::array<double,3>{0,.08,.17}:std::array<double,3>{0,0,0};return c;
}
Detection project(const RigidEstimatorConfig& c,cv::Vec3d center,double phase,int slot){
  const double a=phase+slot*2*CV_PI/3;
  const cv::Vec3d radial(std::cos(a),std::sin(a),0),tangent(-std::sin(a),std::cos(a),0),
    up=std::sin(c.pitch_rad)*radial+cv::Vec3d(0,0,std::cos(c.pitch_rad));
  const cv::Matx33d basis(c.camera_from_target_rotation);std::vector<cv::Point3d> xyz;
  for(auto sign:std::array<cv::Point2d,4>{{{-1,1},{1,1},{1,-1},{-1,-1}}}){
    const auto p=basis*(center+c.radius_m*radial+cv::Vec3d(0,0,c.slot_heights_m[slot])+
      sign.x*c.lightbar_center_distance_m*.5*tangent+sign.y*c.lightbar_length_m*.5*up);
    xyz.emplace_back(p[0],p[1],p[2]);}
  std::vector<cv::Point2d> pixels;cv::projectPoints(xyz,cv::Vec3d(),cv::Vec3d(),c.camera_matrix,c.distortion,pixels);
  Detection d;d.center={0,0};for(int k=0;k<4;++k){d.measurement_points[k]=pixels[k];d.corners[k]=pixels[k];d.center+=d.measurement_points[k]*.25f;}return d;
}
struct Metrics{int frames=0,accepted=0,valid=0,corners=0,wrong_slots=0,generations=0;double squared=0,max_error=0;};
void sweep(){
  std::vector<double> baseline_errors(4800*12,std::numeric_limits<double>::quiet_NaN());
  // Paired runs share identical pseudo-random observations. Offsets encode
  // calibration mismatch only in the estimator, never in the truth generator.
  for(bool covariance:{false,true}){Metrics total;int paired=0,gls_used=0,scalar_used=0;double paired_old=0,paired_new=0;
    for(double distance:{2.,4.})for(double start:{.2,.85})for(double noise:{0.,.15})
      for(bool staggered:{false,true})for(double mismatch:{0.,.01}){
        const auto before=total;
        const auto truth=camera(staggered);auto fit=camera(staggered);fit.covariance_estimation_enabled=covariance;
        fit.camera_matrix=fit.camera_matrix.clone();fit.camera_matrix.at<double>(0,0)*=1+mismatch;
        fit.camera_matrix.at<double>(1,1)*=1+mismatch;fit.radius_m*=1+mismatch;
        CalibratedRigidEstimator estimator(fit);cv::RNG rng(70419);std::uint64_t generation=0;
        for(int i=0;i<150;++i){const double t=i/30.,phase=start+1.1*t;
          const cv::Vec3d center(-distance,.02*t,.2);int visible=0;double best=10;
          for(int s=0;s<3;++s){const double angle=std::abs(std::remainder(phase+s*2*CV_PI/3,2*CV_PI));if(angle<best){best=angle;visible=s;}}
          auto d=project(truth,center,phase,visible);d.center={0,0};
          for(auto& p:d.measurement_points){p.x+=float(rng.gaussian(noise));p.y+=float(rng.gaussian(noise));d.center+=p*.25f;}d.corners=d.measurement_points;
          std::vector<Detection> observations;if(i%47!=20&&i%47!=21)observations.push_back(d);
          if(i%53==25){auto bad=d;bad.measurement_points.fill({10,10});observations.push_back(bad);}
          const auto r=estimator.update(t,observations,.1);++total.frames;total.accepted+=r.measurement_used;total.valid+=r.valid;
          if(r.initialized){
            check(r.fit_method!="none"&&r.fit_selection_reason!="not_fitted","missing selected-branch fit diagnostics");
            if(r.fit_method=="covariance_gls"){
              ++gls_used;check(std::isfinite(r.gls_fit_score)&&r.gls_fit_score<=r.scalar_fit_score+1e-10,"accepted GLS violates residual guard");
            }else ++scalar_used;
          }
          if(r.generation>generation){++total.generations;generation=r.generation;}
          if(r.valid&&r.measurement_used&&staggered)total.wrong_slots+=!r.slots[visible].observed;
          for(int s=0;s<3;++s)if(r.slots[s].future_valid){const auto expected=project(truth,center+cv::Vec3d(0,.002,0),phase+.11,s);
            for(int k=0;k<4;++k){const double error=cv::norm(expected.measurement_points[k]-r.slots[s].future_points[k]);
              check(std::isfinite(error),"nonfinite sweep forecast");++total.corners;total.squared+=error*error;total.max_error=std::max(total.max_error,error);
              auto& previous=baseline_errors[size_t((total.frames-1)*12+s*4+k)];
              if(!covariance)previous=error;
              else if(std::isfinite(previous)){++paired;paired_old+=previous*previous;paired_new+=error*error;}}}
        }
        const int count=total.corners-before.corners;
        std::cout<<"scenario covariance="<<covariance<<" distance="<<distance<<" phase="<<start
          <<" noise="<<noise<<" staggered="<<staggered<<" mismatch="<<mismatch
          <<" valid="<<total.valid-before.valid<<" corners="<<count<<" rms="
          <<(count?std::sqrt((total.squared-before.squared)/count):NAN)<<'\n';
        // Distant/noisy staggered observations have genuinely unresolved intervals.
        // Keep a separate floor from the tighter aggregate and paired budgets.
        check(total.valid-before.valid>=60&&count>=720,"individual scenario forecast coverage regression");
        check(std::sqrt((total.squared-before.squared)/count)<8.,"individual scenario forecast RMS regression");
      }
    check(total.frames==4800&&total.corners>1000,"parameter sweep silently lost denominator/coverage");
    check(total.wrong_slots==0,"parameter sweep swapped resolved staggered slot identities");
    // Fixed acceptance budgets, not recomputed from the implementation under test.
    check(total.valid>=4000&&total.corners>=48000,"aggregate coverage regression");
    check(std::sqrt(total.squared/total.corners)<1.85&&total.max_error<16.,"aggregate forecast error regression");
    check(total.generations==32,"unexpected reacquisition within synthetic scenarios");
    check(scalar_used>0&&(!covariance||gls_used>0),"fit policy diagnostic coverage missing");
    if(covariance)check(paired>=48000&&paired_new<=paired_old*1.03*1.03,"paired forecast error/coverage regression");
    std::cout<<"fit_diagnostics gls="<<gls_used<<" scalar="<<scalar_used<<'\n';
    std::cout<<std::setprecision(10)<<"sweep covariance="<<covariance<<" frames="<<total.frames<<" accepted="<<total.accepted
      <<" valid="<<total.valid<<" forecast_corners="<<total.corners<<" rms_px="<<std::sqrt(total.squared/total.corners)
      <<" max_px="<<total.max_error<<" wrong_slot_frames="<<total.wrong_slots<<" acquisitions="<<total.generations<<'\n';
    if(covariance)std::cout<<"paired_corners="<<paired<<" baseline_rms_px="<<std::sqrt(paired_old/paired)
      <<" covariance_rms_px="<<std::sqrt(paired_new/paired)<<" added_corners="<<total.corners-paired<<'\n';
  }
}
int main(){try{covariance_contract();shape_contract();sweep();std::cout<<"covariance, shape and parameter-sweep contracts passed\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
