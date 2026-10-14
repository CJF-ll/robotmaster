#pragma once
#include <yaml-cpp/yaml.h>
#include <set>
#include <string>
#include <stdexcept>
#include <initializer_list>
namespace outpost {
inline void config_keys(const YAML::Node& node,const std::string& path,
                        std::initializer_list<const char*> allowed,bool optional=false){
  if(!node&&optional)return;
  if(!node.IsMap())throw std::runtime_error(path+" must be a mapping");
  std::set<std::string> names,seen;for(auto name:allowed)names.insert(name);
  for(auto item:node){
    if(!item.first.IsScalar())throw std::runtime_error(path+" contains a non-scalar key");
    const auto key=item.first.as<std::string>();const auto full=path=="root"?key:path+"."+key;
    if(!seen.insert(key).second)throw std::runtime_error("duplicate config key: "+full);
    if(!names.count(key))throw std::runtime_error("unknown config key: "+full);
  }
}
inline void validate_config_keys(const YAML::Node& root){
  config_keys(root,"root",{"schema_version","input","display","detector","prediction","tracking","geometry","rigid_estimator","target_slots","visibility","camera","control","ballistics","rule_speed_prior"});
  config_keys(root["rigid_estimator"],"rigid_estimator",{"history_seconds","min_history_seconds","coast_seconds","min_samples","max_reprojection_error_px","max_orientation_error_rad","max_center_error_m","max_phase_error_rad","max_angular_speed_rad_s","max_center_speed_m_s","min_quad_area_px","identity_score_margin","pose_ambiguity_margin","max_fit_center_rms_m","max_fit_phase_rms_rad","pose_refinement_iterations","pixel_noise_std_px","min_geometry_information_ratio","max_center_std_m","max_phase_std_rad","translation_acceleration_std_m_s2","angular_acceleration_std_rad_s2","max_future_corner_std_px","adaptive_window_enabled","motion_change_threshold_rad","motion_change_threshold_m","motion_change_confirmations","covariance_estimation_enabled","association_nis_scale","robust_fit_gate_std"},true);
  config_keys(root["input"],"input",{"kind","path","timestamp_file","prefetch_frames"});
  config_keys(root["display"],"display",{"enabled"});
  config_keys(root["prediction"],"prediction",{"horizon_seconds"});
  config_keys(root["target_slots"],"target_slots",{"enabled","count","local_observations","profile_samples","sequence_transitions","height_gate_lengths","length_ratio","angle_degrees","ambiguity_margin","reset_gap_seconds"});
  config_keys(root["visibility"],"visibility",{"side_distance_ratio","angle_degrees","length_ratio","min_axial_overlap"});
  config_keys(root["geometry"],"geometry",{"mode","undistort","allow_uniform_resize","calibration_width","calibration_height","camera_matrix","distortion","slot_height_offsets_m","camera_from_target_rotation","armor_width_m","armor_height_m","lightbar_length_m","lightbar_center_distance_m","rotation_radius_m","armor_pitch_deg"});
  const auto detector=root["detector"];
  config_keys(detector,"detector",{"backend","neural","thin_vertical_enabled","thin_min_pixels","reject_border_contours","border_margin_px","brightness","red_difference","min_area","min_length","max_length","min_aspect","max_tilt_deg","max_length_ratio","max_angle_diff_deg","min_separation","max_separation","max_vertical_offset","end_margin_ratio","side_margin_ratio","end_margin_min_px","side_margin_min_px","pairing_ambiguity_margin","endpoint_refinement"});
  config_keys(detector["endpoint_refinement"],"detector.endpoint_refinement",{"enabled","search_fraction","min_contrast","max_spread_px"},true);
  config_keys(detector["neural"],"detector.neural",{"model","classifier_model","score_threshold","nms_threshold","classifier_threshold","crop_axis_half_scale","support_endpoint_fraction","target_class_index","inference_threads","roi_refinement","parallel_branches"},true);
  if(detector["neural"])config_keys(detector["neural"]["roi_refinement"],"detector.neural.roi_refinement",{"enabled","brightness","padding_lengths"},true);
  const auto tracking=root["tracking"];
  config_keys(tracking,"tracking",{"coast_seconds","health","one_sided_guard","common_endpoint_guard","acceleration_noise","measurement_noise","measurement_quality","clean_recovery","low_resolution_noise_scale","initial_velocity_variance","association_pixels","innovation_gate","max_shape_ratio","max_position_std","association_angle_degrees","association_length_ratio","ambiguity_margin","association_miss_cost","confirmation_hits","prediction_hits","shape_guard","center_forecast","coaxial_phase"});
  config_keys(tracking["coaxial_phase"],"tracking.coaxial_phase",{"enabled","window_seconds","fit_interval_seconds","min_omega","max_omega","omega_step","max_fit_rms","ambiguity_ratio","innovation_pixels","max_phase_correction","coast_seconds","stop_omega","min_samples","min_slot_samples","phase_samples","max_samples","offset_gain","association_pixels","association_margin"},true);
  config_keys(tracking["center_forecast"],"tracking.center_forecast",{"enabled","samples","min_span_seconds","max_gap_seconds","blend"},true);
  config_keys(tracking["shape_guard"],"tracking.shape_guard",{"enabled","max_drift_px","history_samples"},true);
  config_keys(tracking["common_endpoint_guard"],"tracking.common_endpoint_guard",{"enabled","shrink_ratio","width_tolerance","max_gap_seconds","min_hits"},true);
  config_keys(tracking["measurement_quality"],"tracking.measurement_quality",{"enabled","max_variance_scale"},true);
  config_keys(tracking["clean_recovery"],"tracking.clean_recovery",{"enabled","max_gap_seconds","max_nis","hits"},true);
  config_keys(tracking["health"],"tracking.health",{"window_seconds","min_samples","nis_warning","failure_fraction"},true);
  config_keys(tracking["one_sided_guard"],"tracking.one_sided_guard",{"enabled","shrink_ratio","other_side_tolerance","min_hits"},true);
  for(auto name:{"camera","control","ballistics","rule_speed_prior"})config_keys(root[name],name,{"enabled"},true);
}
}
