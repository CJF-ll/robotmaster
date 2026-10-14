#pragma once
#include "core.hpp"
#include "run_output.hpp"
#include <opencv2/imgproc.hpp>
#include <fstream>

namespace outpost {
// Only mode-specific serialization/rendering belongs here. Video I/O, detector,
// configuration audit, timing and completion checks live in the shared runner.
class RigidOutput {
 public:
  explicit RigidOutput(const RunOutput& run,const char* time_basis="frame_index/nominal_fps")
    :states_(run.path("rigid_state.csv")),slots_(run.path("rigid_slots.csv")),
     poses_(run.path("pose_measurements.csv")){
    for(auto* s:{&states_,&slots_,&poses_}){
      if(!*s)throw std::runtime_error("cannot create rigid diagnostics");
      s->precision(17);
    }
    states_<<"frame,time,epoch,state,valid,measurement_used,motion_confirmed,identity_resolved,relative_identity_only,cx,cy,cz,vx,vy,vz,phase,omega,age_seconds,fit_center_rms_m,fit_phase_rms_rad,sample_count,generation,association_ambiguous,uncertainty_available,state_reliable,center_std_m,phase_std_rad,omega_std_rad_s,active_history_seconds,adaptive_window_active,fit_method,fit_selection_reason,scalar_fit_score,gls_fit_score\n";
    slots_<<"frame,time,epoch,slot,observed,current_valid,future_valid,detection_index,c0x,c0y,c1x,c1y,c2x,c2y,c3x,c3y,f0x,f0y,f1x,f1y,f2x,f2y,f3x,f3y,future_geometry_valid,prediction_reliable,max_future_corner_std_px,corner0_std_px,corner1_std_px,corner2_std_px,corner3_std_px,future_uncertainty_available,forecast_block_reason\n";
    poses_<<"frame,candidate,slot,accepted,reprojection_error_px,orientation_error_rad,reason,geometry_information_ratio,center_std_m,phase_std_rad\n";
    std::ofstream semantics(run.path("coordinate_semantics.txt"));
    semantics<<"state_basis=target_axis_basis_at_camera_origin\ngeometry=projected_lightbar_centerlines\n"
      "equal_height_ids=relative_to_first_observation\nhidden_slots=model_predictions_not_observations\n"
      "time_basis="<<time_basis<<"\nuncertainty=local_linear_model_not_calibrated_probability\n";
    semantics.flush();if(!semantics)throw std::runtime_error("cannot save coordinate semantics");
  }
  std::pair<size_t,size_t> record_and_draw(Frame& frame,const RigidEstimate& result,
                                          int epoch,const std::vector<Detection>& detections){
    states_<<frame.index<<','<<frame.time_seconds<<','<<epoch<<','<<state_name(result.state)<<','
      <<result.valid<<','<<result.measurement_used<<','<<result.motion_confirmed<<','<<result.identity_resolved<<','
      <<result.relative_identity_only;
    for(double v:result.center.val)states_<<','<<v;
    for(double v:result.velocity.val)states_<<','<<v;
    states_<<','<<result.phase<<','<<result.omega<<','<<result.age_seconds<<','<<result.fit_center_rms_m
      <<','<<result.fit_phase_rms_rad<<','<<result.sample_count<<','<<result.generation<<','<<result.association_ambiguous
      <<','<<result.uncertainty_available<<','<<result.state_reliable<<','<<result.center_std_m
      <<','<<result.phase_std_rad<<','<<result.omega_std_rad_s<<','<<result.active_history_seconds
      <<','<<result.adaptive_window_active<<','<<result.fit_method<<','<<result.fit_selection_reason
      <<','<<result.scalar_fit_score<<','<<result.gls_fit_score<<'\n';
    size_t current=0,future=0;
    auto draw=[&](const auto& quad,cv::Scalar color){
      std::vector<cv::Point> points;for(auto p:quad)points.emplace_back(cvRound(p.x),cvRound(p.y));
      cv::polylines(frame.image,points,true,color,1,cv::LINE_AA);
    };
    for(const auto& slot:result.slots){
      slots_<<frame.index<<','<<frame.time_seconds<<','<<epoch<<','<<slot.id<<','<<slot.observed<<','
        <<slot.current_valid<<','<<slot.future_valid<<','<<slot.detection_index;
      for(auto p:slot.current_points)slots_<<','<<p.x<<','<<p.y;
      for(auto p:slot.future_points)slots_<<','<<p.x<<','<<p.y;
      slots_<<','<<slot.future_geometry_valid<<','<<slot.prediction_reliable<<','<<slot.max_future_corner_std_px;
      for(double value:slot.future_corner_std_px)slots_<<','<<value;
      slots_<<','<<slot.future_uncertainty_available<<','<<slot.forecast_block_reason<<'\n';
      if(slot.observed&&slot.detection_index>=0&&size_t(slot.detection_index)<detections.size()){
        const auto& quad=detections[size_t(slot.detection_index)].measurement_points;
        draw(quad,{0,255,0});++current;
        const auto label=result.identity_resolved&&!result.association_ambiguous?
          "A"+std::to_string(slot.id+1)+" obs":"obs / identity unresolved";
        cv::putText(frame.image,label,quad[0],cv::FONT_HERSHEY_SIMPLEX,.35,{0,255,0},1);
      }else if(result.valid&&slot.current_valid){draw(slot.current_points,{0,180,255});++current;}
      if(slot.future_valid){draw(slot.future_points,{0,0,255});++future;}
    }
    for(const auto& d:result.diagnostics)poses_<<frame.index<<','<<d.detection_index<<','<<d.slot<<','<<d.accepted
      <<','<<d.reprojection_error_px<<','<<d.orientation_error_rad<<','<<d.reason
      <<','<<d.geometry_information_ratio<<','<<d.center_std_m<<','<<d.phase_std_rad<<'\n';
    cv::putText(frame.image,"GREEN=observed ORANGE=model RED=future (lamp centerlines)",
      {8,22},cv::FONT_HERSHEY_SIMPLEX,.35,{255,255,255},1);
    const std::string status=std::string(state_name(result.state))+" omega="+
      (result.valid?std::to_string(result.omega):"unavailable")+" rad/s";
    cv::putText(frame.image,status,{8,42},cv::FONT_HERSHEY_SIMPLEX,.4,{255,255,255},1);
    if(result.relative_identity_only)cv::putText(frame.image,"IDs are relative; no permanent physical label",
      {8,62},cv::FONT_HERSHEY_SIMPLEX,.35,{0,220,255},1);
    return {current,future};
  }
  void finish(){for(auto* s:{&states_,&slots_,&poses_}){s->flush();if(!*s)throw std::runtime_error("incomplete rigid log");}}
 private:
  static const char* state_name(RigidTrackingState state){
    switch(state){case RigidTrackingState::Lost:return "lost";
      case RigidTrackingState::Initializing:return "initializing";
      case RigidTrackingState::Tracking:return "tracking";
      case RigidTrackingState::Coasting:return "coasting";}
    return "unknown";
  }
  std::ofstream states_,slots_,poses_;
};
}
