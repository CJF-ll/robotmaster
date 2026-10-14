#include "core.hpp"
#include "config_schema.hpp"
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

namespace outpost {
RigidEstimatorConfig calibrated_config(const Config& config,cv::Size size){
  const auto& g=config.geometry;auto r=config.rigid;
  if(size.width<=0||size.height<=0||g.calibration_width<=0||g.calibration_height<=0||
     g.camera_matrix.size()!=9||g.distortion.empty()||g.camera_from_target_rotation.size()!=9||
     !g.lightbar_length_m||!g.lightbar_center_distance_m||!g.rotation_radius_m||
     !g.armor_pitch_deg||g.slot_height_offsets_m.size()!=3)
    throw std::runtime_error("calibrated estimator requires complete measured geometry");
  const double sx=double(size.width)/g.calibration_width,sy=double(size.height)/g.calibration_height;
  if(size!=cv::Size(g.calibration_width,g.calibration_height)&&
     (!g.allow_uniform_resize||std::abs(sx-sy)>1e-9))
    throw std::runtime_error("calibration/video resolution mismatch");
  r.camera_matrix=cv::Mat(3,3,CV_64F,const_cast<double*>(g.camera_matrix.data())).clone();
  r.camera_matrix.row(0)*=sx;r.camera_matrix.row(1)*=sy;
  r.distortion=g.undistort?cv::Mat::zeros(1,5,CV_64F):cv::Mat(g.distortion,true);
  r.camera_from_target_rotation=cv::Mat(3,3,CV_64F,const_cast<double*>(g.camera_from_target_rotation.data())).clone();
  r.lightbar_length_m=*g.lightbar_length_m;r.lightbar_center_distance_m=*g.lightbar_center_distance_m;
  r.radius_m=*g.rotation_radius_m;r.pitch_rad=*g.armor_pitch_deg*CV_PI/180.;
  std::copy(g.slot_height_offsets_m.begin(),g.slot_height_offsets_m.end(),r.slot_heights_m.begin());
  return r;
}
Config Config::load(const std::string& path) {
  std::ifstream file(path);if(!file)throw std::runtime_error("cannot read config: "+path);
  std::ostringstream data;data<<file.rdbuf();return parse(data.str());
}
Config Config::parse(const std::string& text) {
  const auto root = YAML::Load(text);
  validate_config_keys(root);
  if (root["schema_version"].as<int>() != 1)
    throw std::runtime_error("unsupported schema_version");
  if (root["input"]["kind"].as<std::string>() != "video")
    throw std::runtime_error("this offline build only accepts video input");
  for (const auto name : {"camera", "control", "ballistics", "rule_speed_prior"})
    if (root[name] && root[name]["enabled"].as<bool>(false))
      throw std::runtime_error(std::string(name) + " is not available in offline build");
  Config c;
  c.timestamp_file=root["input"]["timestamp_file"].as<std::string>("");
  c.prefetch_frames=root["input"]["prefetch_frames"].as<int>(0);
  if(c.prefetch_frames<0||c.prefetch_frames>8)throw std::runtime_error("prefetch_frames must be 0..8");
  c.geometry.mode=root["geometry"]["mode"].as<std::string>();
  if(c.geometry.mode!="image_2d"&&c.geometry.mode!="calibrated_3d")
    throw std::runtime_error("unknown geometry mode");
  c.source_text=text;
  const auto rigid=root["rigid_estimator"];
  if(rigid){
#define READ_RIGID(field) c.rigid.field=rigid[#field].as<decltype(c.rigid.field)>(c.rigid.field)
    READ_RIGID(covariance_estimation_enabled);READ_RIGID(association_nis_scale);READ_RIGID(robust_fit_gate_std);
    READ_RIGID(history_seconds);READ_RIGID(min_history_seconds);READ_RIGID(coast_seconds);
    READ_RIGID(min_samples);READ_RIGID(max_reprojection_error_px);READ_RIGID(max_orientation_error_rad);
    READ_RIGID(max_center_error_m);READ_RIGID(max_phase_error_rad);READ_RIGID(max_angular_speed_rad_s);
    READ_RIGID(max_center_speed_m_s);READ_RIGID(min_quad_area_px);READ_RIGID(identity_score_margin);
    READ_RIGID(pose_ambiguity_margin);READ_RIGID(max_fit_center_rms_m);READ_RIGID(max_fit_phase_rms_rad);
    READ_RIGID(pose_refinement_iterations);READ_RIGID(pixel_noise_std_px);READ_RIGID(min_geometry_information_ratio);
    READ_RIGID(max_center_std_m);READ_RIGID(max_phase_std_rad);READ_RIGID(translation_acceleration_std_m_s2);
    READ_RIGID(angular_acceleration_std_rad_s2);READ_RIGID(max_future_corner_std_px);READ_RIGID(adaptive_window_enabled);
    READ_RIGID(motion_change_threshold_rad);READ_RIGID(motion_change_threshold_m);READ_RIGID(motion_change_confirmations);
#undef READ_RIGID
  }
  const auto& rc=c.rigid;
  for(double v:{rc.history_seconds,rc.min_history_seconds,rc.coast_seconds,rc.max_reprojection_error_px,
      rc.max_orientation_error_rad,rc.max_center_error_m,rc.max_phase_error_rad,rc.max_angular_speed_rad_s,
      rc.max_center_speed_m_s,rc.min_quad_area_px,rc.identity_score_margin,rc.pose_ambiguity_margin,
      rc.max_fit_center_rms_m,rc.max_fit_phase_rms_rad,rc.pixel_noise_std_px,rc.min_geometry_information_ratio,
      rc.max_center_std_m,rc.max_phase_std_rad,rc.translation_acceleration_std_m_s2,
      rc.angular_acceleration_std_rad_s2,rc.max_future_corner_std_px,rc.motion_change_threshold_rad,rc.motion_change_threshold_m,
      rc.association_nis_scale,rc.robust_fit_gate_std})
    if(!std::isfinite(v)||v<=0)throw std::runtime_error("invalid rigid estimator settings");
  if(rc.min_samples<3||rc.min_samples>10000||rc.min_history_seconds>rc.history_seconds||
     rc.history_seconds>10||rc.coast_seconds>5||rc.max_orientation_error_rad>=CV_PI/2||rc.max_phase_error_rad>=CV_PI/3||
     rc.pose_refinement_iterations<1||rc.pose_refinement_iterations>100||rc.min_geometry_information_ratio>=1||
     rc.max_phase_std_rad>=CV_PI||rc.motion_change_confirmations<2||rc.motion_change_confirmations>100)
    throw std::runtime_error("inconsistent rigid estimator settings");
  const auto slots=root["target_slots"];
  c.slots.enabled=slots["enabled"].as<bool>();
  c.slots.count=slots["count"].as<int>();
  c.slots.local_observations=slots["local_observations"].as<int>();
  c.slots.profile_samples=slots["profile_samples"].as<int>();
  c.slots.sequence_transitions=slots["sequence_transitions"].as<int>();
  c.slots.height_gate_lengths=slots["height_gate_lengths"].as<double>();
  c.slots.length_ratio=slots["length_ratio"].as<double>();
  c.slots.angle_degrees=slots["angle_degrees"].as<double>();
  c.slots.ambiguity_margin=slots["ambiguity_margin"].as<double>();
  c.slots.reset_gap_seconds=slots["reset_gap_seconds"].as<double>();
  if(c.slots.count<2||c.slots.count>8||c.slots.local_observations<1||c.slots.profile_samples<7||
     c.slots.sequence_transitions<2*c.slots.count)
    throw std::runtime_error("invalid slot confirmation history");
  for(double value:{c.slots.height_gate_lengths,c.slots.length_ratio,c.slots.angle_degrees,
      c.slots.ambiguity_margin,c.slots.reset_gap_seconds})
    if(!std::isfinite(value)||value<=0)throw std::runtime_error("invalid slot gate");
  if(c.slots.length_ratio<1||c.slots.angle_degrees>90)
    throw std::runtime_error("invalid slot geometry gate");
  const auto geometry=root["geometry"];
  c.geometry.undistort=geometry["undistort"].as<bool>(false);
  c.geometry.allow_uniform_resize=geometry["allow_uniform_resize"].as<bool>(false);
  c.geometry.calibration_width=geometry["calibration_width"].as<int>(0);
  c.geometry.calibration_height=geometry["calibration_height"].as<int>(0);
  auto vector_value=[&](const char* key,std::vector<double>& target){
    if(geometry[key]&&!geometry[key].IsNull())target=geometry[key].as<std::vector<double>>();
    for(double v:target)if(!std::isfinite(v))throw std::runtime_error("nonfinite geometry");
  };
  vector_value("camera_matrix",c.geometry.camera_matrix);
  vector_value("distortion",c.geometry.distortion);
  vector_value("slot_height_offsets_m",c.geometry.slot_height_offsets_m);
  vector_value("camera_from_target_rotation",c.geometry.camera_from_target_rotation);
  if(!c.geometry.camera_from_target_rotation.empty()){
    if(c.geometry.camera_from_target_rotation.size()!=9)throw std::runtime_error("rotation requires 9 values");
    cv::Mat r(3,3,CV_64F,c.geometry.camera_from_target_rotation.data());
    if(cv::norm(r*r.t()-cv::Mat::eye(3,3,CV_64F))>1e-6||std::abs(cv::determinant(r)-1)>1e-6)
      throw std::runtime_error("camera_from_target_rotation must be a proper rotation");
  }
  const auto& k=c.geometry.camera_matrix;const auto& dist=c.geometry.distortion;
  if(!k.empty()&&(k.size()!=9||k[0]<=0||k[4]<=0||k[1]!=0||k[3]!=0||k[6]!=0||k[7]!=0||k[8]!=1))
    throw std::runtime_error("invalid camera matrix");
  if(!dist.empty()&&dist.size()!=4&&dist.size()!=5&&dist.size()!=8&&dist.size()!=12&&dist.size()!=14)
    throw std::runtime_error("invalid distortion coefficient count");
  if(c.geometry.calibration_width<0||c.geometry.calibration_height<0||
     ((c.geometry.calibration_width==0)!=(c.geometry.calibration_height==0)))
    throw std::runtime_error("invalid calibration reference resolution");
  if(c.geometry.undistort&&(k.empty()||dist.empty()||c.geometry.calibration_width==0))
    throw std::runtime_error("undistortion requires measured K, distortion and calibration resolution");
  auto scalar=[&](const char* key,std::optional<double>& target,bool positive){
    if(geometry[key]&&!geometry[key].IsNull()){
      double v=geometry[key].as<double>();
      if(!std::isfinite(v)||(positive&&v<=0))
        throw std::runtime_error("invalid physical geometry");
      target=v;
    }
  };
  scalar("armor_width_m",c.geometry.armor_width_m,true);
  scalar("armor_height_m",c.geometry.armor_height_m,true);
  scalar("lightbar_length_m",c.geometry.lightbar_length_m,true);
  scalar("lightbar_center_distance_m",c.geometry.lightbar_center_distance_m,true);
  scalar("rotation_radius_m",c.geometry.rotation_radius_m,true);
  scalar("armor_pitch_deg",c.geometry.armor_pitch_deg,false);
  if(c.geometry.mode=="calibrated_3d"&&(k.empty()||dist.empty()||
     c.geometry.calibration_width<=0||c.geometry.calibration_height<=0||
     c.geometry.camera_from_target_rotation.empty()||!c.geometry.lightbar_length_m||
     !c.geometry.lightbar_center_distance_m||!c.geometry.rotation_radius_m||
     !c.geometry.armor_pitch_deg||c.geometry.slot_height_offsets_m.size()!=3))
    throw std::runtime_error("calibrated_3d requires measured K/D/resolution, axis rotation, lamp dimensions, radius, pitch and three heights");
  const auto vis=root["visibility"];
  c.visibility.side_distance_ratio=vis["side_distance_ratio"].as<double>();
  c.visibility.angle_degrees=vis["angle_degrees"].as<double>();
  c.visibility.length_ratio=vis["length_ratio"].as<double>();
  c.visibility.min_axial_overlap=vis["min_axial_overlap"].as<double>(.6);
  const auto& v=c.visibility;
  if(!std::isfinite(v.side_distance_ratio)||v.side_distance_ratio<=0||v.side_distance_ratio>1||
     !std::isfinite(v.angle_degrees)||v.angle_degrees<0||v.angle_degrees>90||
     !std::isfinite(v.length_ratio)||v.length_ratio<1||
     !std::isfinite(v.min_axial_overlap)||v.min_axial_overlap<0||v.min_axial_overlap>1)
    throw std::runtime_error("invalid visibility geometry");
  const auto det=root["detector"];
  c.detector.backend=det["backend"].as<std::string>("traditional");
  if(c.detector.backend!="traditional"&&c.detector.backend!="neural_supported")throw std::runtime_error("unknown detector backend");
#ifndef OUTPOST_OPENVINO_ENABLED
  if(c.detector.backend!="traditional")throw std::runtime_error("requested detector backend is not built; no fallback");
#endif
  const auto neural=det["neural"];
  if(neural){auto& n=c.detector;
    n.neural_model=neural["model"].as<std::string>("");
    n.classifier_model=neural["classifier_model"].as<std::string>("");
    n.neural_score=neural["score_threshold"].as<double>(.7);
    n.neural_nms=neural["nms_threshold"].as<double>(.3);
    n.classifier_threshold=neural["classifier_threshold"].as<double>(.7);
    n.classifier_crop_half_scale=neural["crop_axis_half_scale"].as<double>(1.125);
    n.support_endpoint_fraction=neural["support_endpoint_fraction"].as<double>(.5);
    n.classifier_target_index=neural["target_class_index"].as<int>(6);
    n.neural_inference_threads=neural["inference_threads"].as<int>(0);
    n.parallel_branches=neural["parallel_branches"].as<bool>(false);
    const auto roi=neural["roi_refinement"];
    if(roi){
      n.roi_refinement_enabled=roi["enabled"].as<bool>(false);
      n.roi_brightness=roi["brightness"].as<int>(90);
      n.roi_padding_lengths=roi["padding_lengths"].as<double>(.6);
    }
  }
  const auto& n=c.detector;
  if(n.neural_inference_threads<0||n.neural_inference_threads>256)
    throw std::runtime_error("inference_threads must be 0 (automatic) or 1..256");
  for(double value:{n.neural_score,n.neural_nms,n.classifier_threshold})
    if(!std::isfinite(value)||value<0||value>1)throw std::runtime_error("invalid neural threshold");
  if(!std::isfinite(n.classifier_crop_half_scale)||n.classifier_crop_half_scale<=0||
     !std::isfinite(n.support_endpoint_fraction)||n.support_endpoint_fraction<=0||n.support_endpoint_fraction>1||
     n.classifier_target_index<0||n.classifier_target_index>=9)throw std::runtime_error("invalid neural geometry/class configuration");
  if(n.backend=="neural_supported"&&(n.neural_model.empty()||n.classifier_model.empty()))
    throw std::runtime_error("neural_supported requires both model paths");
  std::ostringstream detector_signature;detector_signature.precision(17);
  detector_signature<<n.backend<<'\n'<<n.neural_model<<'\n'<<n.classifier_model<<'\n'
    <<n.neural_score<<','<<n.neural_nms<<','<<n.classifier_threshold<<','<<n.classifier_crop_half_scale
    <<','<<n.support_endpoint_fraction<<','<<n.classifier_target_index<<','<<n.neural_inference_threads
    <<','<<n.roi_refinement_enabled<<','<<n.roi_brightness<<','<<n.roi_padding_lengths<<','<<n.parallel_branches;
  c.detector_runtime_signature=detector_signature.str();
  c.detector.thin_vertical_enabled=det["thin_vertical_enabled"].as<bool>(true);
  c.detector.thin_min_pixels=det["thin_min_pixels"].as<int>(8);
  if(c.detector.thin_min_pixels<3)throw std::runtime_error("thin_min_pixels must be at least 3");
  c.detector.reject_border_contours=det["reject_border_contours"].as<bool>(true);
  c.detector.border_margin_px=det["border_margin_px"].as<int>(0);
  if(c.detector.border_margin_px<0)throw std::runtime_error("negative image border margin");
  c.detector.brightness=det["brightness"].as<int>();
  c.detector.red_difference=det["red_difference"].as<int>();
  c.detector.min_area=det["min_area"].as<double>();
  c.detector.min_length=det["min_length"].as<double>();
  c.detector.max_length=det["max_length"].as<double>();
  c.detector.min_aspect=det["min_aspect"].as<double>();
  c.detector.max_tilt_deg=det["max_tilt_deg"].as<double>();
  c.detector.max_length_ratio=det["max_length_ratio"].as<double>();
  c.detector.max_angle_diff_deg=det["max_angle_diff_deg"].as<double>();
  c.detector.min_separation=det["min_separation"].as<double>();
  c.detector.max_separation=det["max_separation"].as<double>();
  c.detector.max_vertical_offset=det["max_vertical_offset"].as<double>();
  c.detector.end_margin_ratio=det["end_margin_ratio"].as<double>();
  c.detector.side_margin_ratio=det["side_margin_ratio"].as<double>();
  c.detector.end_margin_min_px=det["end_margin_min_px"].as<double>(1);
  c.detector.side_margin_min_px=det["side_margin_min_px"].as<double>(.75);
  c.detector.pairing_ambiguity_margin=det["pairing_ambiguity_margin"].as<double>(.05);
  const auto endpoint=det["endpoint_refinement"];
  if(endpoint){
    c.detector.endpoint_refinement_enabled=endpoint["enabled"].as<bool>(false);
    c.detector.endpoint_search_fraction=endpoint["search_fraction"].as<double>(.35);
    c.detector.endpoint_min_contrast=endpoint["min_contrast"].as<double>(8.);
    c.detector.endpoint_max_spread_px=endpoint["max_spread_px"].as<double>(1.5);
  }
  const auto& d=c.detector;
  if(!std::isfinite(d.endpoint_search_fraction)||d.endpoint_search_fraction<=0||d.endpoint_search_fraction>1||
     !std::isfinite(d.endpoint_min_contrast)||d.endpoint_min_contrast<=0||d.endpoint_min_contrast>255||
     !std::isfinite(d.endpoint_max_spread_px)||d.endpoint_max_spread_px<=0||d.endpoint_max_spread_px>5)
    throw std::runtime_error("invalid endpoint refinement settings");
  if(d.roi_brightness<1||d.roi_brightness>255||
     (d.roi_refinement_enabled&&d.roi_brightness>d.brightness)||
     !std::isfinite(d.roi_padding_lengths)||d.roi_padding_lengths<=0||d.roi_padding_lengths>2)
    throw std::runtime_error("invalid local ROI refinement settings");
  if(d.brightness<0 || d.brightness>255 || d.red_difference<0 || d.red_difference>255)
    throw std::runtime_error("invalid detector threshold");
  for(double value : {d.min_area,d.min_length,d.max_length,d.min_aspect,d.max_tilt_deg,
      d.max_length_ratio,d.max_angle_diff_deg,d.min_separation,d.max_separation,
      d.max_vertical_offset,d.end_margin_ratio,d.side_margin_ratio,d.end_margin_min_px,d.side_margin_min_px,d.pairing_ambiguity_margin})
    if(!std::isfinite(value)||value<0) throw std::runtime_error("invalid detector geometry");
  if(d.min_length<=0 || d.max_length<d.min_length || d.min_aspect<1 ||
     d.max_length_ratio<1 || d.max_tilt_deg>=85 || d.max_angle_diff_deg>90 ||
     d.max_separation<=d.min_separation)
    throw std::runtime_error("inconsistent detector geometry");
  c.input = root["input"]["path"].as<std::string>();
  c.prediction_seconds = root["prediction"]["horizon_seconds"].as<double>();
  c.coast_seconds = root["tracking"]["coast_seconds"].as<double>();
  auto& t=c.tracker;t.coast_seconds=c.coast_seconds;
  const auto tr=root["tracking"];
  const auto shape=tr["shape_guard"];
  if(shape){t.shape_guard_enabled=shape["enabled"].as<bool>(false);
    t.max_shape_drift_px=shape["max_drift_px"].as<double>(3.);
    t.shape_history_samples=shape["history_samples"].as<int>(4);}
  if(!std::isfinite(t.max_shape_drift_px)||t.max_shape_drift_px<=0||t.shape_history_samples<3||t.shape_history_samples>30)
    throw std::runtime_error("invalid shape guard settings");
  const auto common=tr["common_endpoint_guard"];
  if(common){
    t.common_endpoint_guard_enabled=common["enabled"].as<bool>(false);
    t.common_endpoint_shrink_ratio=common["shrink_ratio"].as<double>(.75);
    t.common_endpoint_width_tolerance=common["width_tolerance"].as<double>(.10);
    t.common_endpoint_max_gap_seconds=common["max_gap_seconds"].as<double>(.10);
    t.common_endpoint_min_hits=common["min_hits"].as<int>(5);
  }
  if(!std::isfinite(t.common_endpoint_shrink_ratio)||t.common_endpoint_shrink_ratio<=0||t.common_endpoint_shrink_ratio>=1||
     !std::isfinite(t.common_endpoint_width_tolerance)||t.common_endpoint_width_tolerance<0||t.common_endpoint_width_tolerance>=.5||
     !std::isfinite(t.common_endpoint_max_gap_seconds)||t.common_endpoint_max_gap_seconds<=0||
     t.common_endpoint_min_hits<2||t.common_endpoint_min_hits>100||
     (t.common_endpoint_guard_enabled&&t.common_endpoint_max_gap_seconds>t.coast_seconds))
    throw std::runtime_error("invalid common endpoint guard settings");
  const auto health=tr["health"];
  if(health){
    t.health.window_seconds=health["window_seconds"].as<double>(.5);
    t.health.min_samples=health["min_samples"].as<int>(8);
    t.health.nis_warning=health["nis_warning"].as<double>(5.991464547);
    t.health.failure_fraction=health["failure_fraction"].as<double>(.4);
  }
  if(!std::isfinite(t.health.window_seconds)||t.health.window_seconds<=0||t.health.window_seconds>10||
     t.health.min_samples<2||t.health.min_samples>10000||!std::isfinite(t.health.nis_warning)||
     t.health.nis_warning<=0||!std::isfinite(t.health.failure_fraction)||
     t.health.failure_fraction<=0||t.health.failure_fraction>1)
    throw std::runtime_error("invalid advisory health settings");
  t.acceleration_noise=tr["acceleration_noise"].as<double>();
  t.measurement_noise=tr["measurement_noise"].as<double>();
  const auto guard=tr["one_sided_guard"];
  if(guard){
    t.one_sided_guard_enabled=guard["enabled"].as<bool>(false);
    t.one_sided_shrink_ratio=guard["shrink_ratio"].as<double>(.7);
    t.other_side_tolerance=guard["other_side_tolerance"].as<double>(.1);
    t.one_sided_min_hits=guard["min_hits"].as<int>(5);
  }
  if(!std::isfinite(t.one_sided_shrink_ratio)||t.one_sided_shrink_ratio<=0||t.one_sided_shrink_ratio>=1||
     !std::isfinite(t.other_side_tolerance)||t.other_side_tolerance<0||t.other_side_tolerance>=1||
     t.one_sided_min_hits<2||t.one_sided_min_hits>10000)
    throw std::runtime_error("invalid one-sided temporal guard");
  t.low_resolution_noise_scale=tr["low_resolution_noise_scale"].as<double>(1);
  if(!std::isfinite(t.low_resolution_noise_scale)||t.low_resolution_noise_scale<1||
     !std::isfinite(t.measurement_noise*t.low_resolution_noise_scale))
    throw std::runtime_error("low_resolution_noise_scale must be finite and at least 1");
  t.initial_velocity_variance=tr["initial_velocity_variance"].as<double>();
  t.association_pixels=tr["association_pixels"].as<double>();
  t.innovation_gate=tr["innovation_gate"].as<double>();
  t.max_shape_ratio=tr["max_shape_ratio"].as<double>();
  t.max_position_std=tr["max_position_std"].as<double>();
  t.association_angle_degrees=tr["association_angle_degrees"].as<double>();
  t.association_length_ratio=tr["association_length_ratio"].as<double>();
  t.ambiguity_margin=tr["ambiguity_margin"].as<double>();
  t.association_miss_cost=tr["association_miss_cost"].as<double>(15);
  t.confirmation_hits=tr["confirmation_hits"].as<int>();
  t.prediction_hits=tr["prediction_hits"].as<int>();
  const auto mq=tr["measurement_quality"];
  if(mq){t.measurement_quality_enabled=mq["enabled"].as<bool>(false);
    t.measurement_quality_max_scale=mq["max_variance_scale"].as<double>(4);}
  if(!std::isfinite(t.measurement_quality_max_scale)||t.measurement_quality_max_scale<1||
     t.measurement_quality_max_scale>100||
     !std::isfinite(t.measurement_noise*t.low_resolution_noise_scale*t.measurement_quality_max_scale))
    throw std::runtime_error("invalid measurement quality variance scale");
  const auto recovery=tr["clean_recovery"];
  if(recovery){
    t.clean_recovery_enabled=recovery["enabled"].as<bool>(false);
    t.clean_recovery_max_gap_seconds=recovery["max_gap_seconds"].as<double>(.1);
    t.clean_recovery_max_nis=recovery["max_nis"].as<double>(4);
    t.clean_recovery_hits=recovery["hits"].as<int>(2);
  }
  if(!std::isfinite(t.clean_recovery_max_gap_seconds)||t.clean_recovery_max_gap_seconds<=0||
     !std::isfinite(t.clean_recovery_max_nis)||t.clean_recovery_max_nis<=0||
     t.clean_recovery_hits<2||t.clean_recovery_hits>10000||
     (t.clean_recovery_enabled&&(t.clean_recovery_max_gap_seconds>t.coast_seconds||
      t.clean_recovery_max_nis>t.innovation_gate||t.clean_recovery_hits>t.prediction_hits)))
    throw std::runtime_error("invalid clean recovery settings");
  for(double value:{t.acceleration_noise,t.measurement_noise,t.initial_velocity_variance,t.association_miss_cost,
      t.association_pixels,t.innovation_gate,t.max_shape_ratio,t.max_position_std,
      t.association_angle_degrees,t.association_length_ratio})
    if(!std::isfinite(value)||value<=0)throw std::runtime_error("invalid tracking noise/gate");
  if(t.max_shape_ratio<1||t.confirmation_hits<1||t.prediction_hits<t.confirmation_hits)
    throw std::runtime_error("invalid track confirmation settings");
  if(t.association_angle_degrees>90||t.association_length_ratio<1||
     !std::isfinite(t.ambiguity_margin)||t.ambiguity_margin<0)
    throw std::runtime_error("invalid association settings");
  c.display = root["display"]["enabled"].as<bool>();
  if (c.input.empty() || !std::isfinite(c.prediction_seconds) ||
      c.prediction_seconds < 0 || c.prediction_seconds > 1 ||
      !std::isfinite(c.coast_seconds) || c.coast_seconds < 0 || c.coast_seconds > 5)
    throw std::runtime_error("invalid input or time settings");
  // Typed signatures describe effective consumers, not all stored YAML fields.
  // Inactive settings remain parsed/validated/snapshotted for later use.
  auto values=[](std::ostream& out,const std::vector<double>& v){out<<v.size()<<':';for(double x:v)out<<x<<',';out<<';';};
  auto camera=[&](std::ostream& out){
    // The input size is immutable within a run. Normalized K is exactly the
    // effective K up to that fixed size, avoiding resets for equivalent scaled
    // calibration representations. Resize permission is validated separately.
    out<<k[0]/c.geometry.calibration_width<<','<<k[2]/c.geometry.calibration_width<<','
       <<k[4]/c.geometry.calibration_height<<','<<k[5]/c.geometry.calibration_height<<';';
  };
  std::ostringstream image_signature;image_signature.precision(17);
  image_signature<<(c.geometry.undistort?"rectified;":"raw;");
  if(c.geometry.undistort){camera(image_signature);values(image_signature,dist);}
  c.preprocessing_signature=image_signature.str();
  std::ostringstream signature;signature.precision(17);
  signature<<c.geometry.mode<<';'<<c.preprocessing_signature;
  signature<<d.thin_vertical_enabled<<';';if(d.thin_vertical_enabled)signature<<d.thin_min_pixels<<';';
  signature<<d.reject_border_contours<<';';if(d.reject_border_contours)signature<<d.border_margin_px<<';';
  for(double value:{d.min_area,d.min_length,d.max_length,d.min_aspect,d.max_tilt_deg,
      d.max_length_ratio,d.max_angle_diff_deg,d.min_separation,d.max_separation,d.max_vertical_offset,
      d.end_margin_ratio,d.side_margin_ratio,d.end_margin_min_px,d.side_margin_min_px,d.pairing_ambiguity_margin})signature<<value<<',';
  signature<<';'<<d.endpoint_refinement_enabled<<';';
  if(d.endpoint_refinement_enabled)signature<<d.endpoint_search_fraction<<','<<d.endpoint_min_contrast<<','<<d.endpoint_max_spread_px<<';';
  if(c.geometry.mode=="image_2d"){
    for(double value:{t.low_resolution_noise_scale,t.acceleration_noise,t.measurement_noise,t.initial_velocity_variance,
        t.association_pixels,t.innovation_gate,t.max_shape_ratio,t.coast_seconds,t.max_position_std,t.association_angle_degrees,
        t.association_length_ratio,t.ambiguity_margin,t.association_miss_cost})signature<<value<<',';
    signature<<t.confirmation_hits<<','<<t.prediction_hits<<';';
    signature<<t.shape_guard_enabled<<';';if(t.shape_guard_enabled)signature<<t.max_shape_drift_px<<','<<t.shape_history_samples<<';';
    signature<<t.measurement_quality_enabled<<';';if(t.measurement_quality_enabled)signature<<t.measurement_quality_max_scale<<';';
    signature<<t.clean_recovery_enabled<<';';if(t.clean_recovery_enabled)signature<<t.clean_recovery_max_gap_seconds<<','<<t.clean_recovery_max_nis<<','<<t.clean_recovery_hits<<';';
    signature<<t.one_sided_guard_enabled<<';';if(t.one_sided_guard_enabled)signature<<t.one_sided_shrink_ratio<<','<<t.other_side_tolerance<<','<<t.one_sided_min_hits<<';';
    signature<<t.common_endpoint_guard_enabled<<';';if(t.common_endpoint_guard_enabled)signature<<t.common_endpoint_shrink_ratio<<','<<t.common_endpoint_width_tolerance<<','<<t.common_endpoint_max_gap_seconds<<','<<t.common_endpoint_min_hits<<';';
    signature<<t.health.window_seconds<<','<<t.health.min_samples<<','<<t.health.nis_warning<<','<<t.health.failure_fraction<<';';
    signature<<c.slots.enabled<<';';if(c.slots.enabled)signature<<c.slots.count<<','<<c.slots.local_observations<<','<<c.slots.profile_samples<<','
      <<c.slots.sequence_transitions<<','<<c.slots.height_gate_lengths<<','<<c.slots.length_ratio<<','
      <<c.slots.angle_degrees<<','<<c.slots.ambiguity_margin<<','<<c.slots.reset_gap_seconds<<';';
  }else{
    camera(signature);if(!c.geometry.undistort)values(signature,dist);
    values(signature,c.geometry.camera_from_target_rotation);values(signature,c.geometry.slot_height_offsets_m);
    signature<<*c.geometry.lightbar_length_m<<','<<*c.geometry.lightbar_center_distance_m<<','
      <<*c.geometry.rotation_radius_m<<','<<*c.geometry.armor_pitch_deg<<';';
    signature<<rc.min_samples<<','<<rc.pose_refinement_iterations<<';';
    signature<<rc.covariance_estimation_enabled<<';';if(rc.covariance_estimation_enabled)
      signature<<rc.association_nis_scale<<','<<rc.robust_fit_gate_std<<';';
    for(double v:{rc.history_seconds,rc.min_history_seconds,rc.coast_seconds,rc.max_reprojection_error_px,
        rc.max_orientation_error_rad,rc.max_center_error_m,rc.max_phase_error_rad,rc.max_angular_speed_rad_s,
        rc.max_center_speed_m_s,rc.min_quad_area_px,rc.identity_score_margin,rc.pose_ambiguity_margin,
        rc.max_fit_center_rms_m,rc.max_fit_phase_rms_rad,rc.pixel_noise_std_px,rc.min_geometry_information_ratio,
        rc.max_center_std_m,rc.max_phase_std_rad,rc.translation_acceleration_std_m_s2,
        rc.angular_acceleration_std_rad_s2,rc.max_future_corner_std_px})signature<<v<<',';
    signature<<';'<<rc.adaptive_window_enabled<<';';if(rc.adaptive_window_enabled)signature
      <<rc.motion_change_threshold_rad<<','<<rc.motion_change_threshold_m<<','<<rc.motion_change_confirmations<<';';
  }
  c.estimation_signature=signature.str();
  if(c.geometry.mode=="calibrated_3d"){
    CalibratedRigidEstimator validate(calibrated_config(c,{c.geometry.calibration_width,c.geometry.calibration_height}));
  }
  return c;
}
ConfigRuntime::ConfigRuntime(const std::string& path):path_(path),current_(Config::load(path)){}
void ConfigRuntime::bind_input_size(cv::Size size){
  if(size==input_size_)return;
  if(input_size_.area()!=0)throw std::runtime_error("video resolution changed; restart required");
  ImagePreprocessor prepared(current_.geometry,size);
  if(current_.geometry.mode=="calibrated_3d")calibrated_config(current_,size);
  preprocessor_=std::move(prepared);input_size_=size;
}
ReloadResult ConfigRuntime::apply_text(const std::string& text,std::string& message){
  message.clear();
  if(has_last_text_&&text==last_text_){
    if(last_rejected_){message=last_rejection_;return ReloadResult::Rejected;}
    return ReloadResult::Unchanged;
  }
  last_text_=text;has_last_text_=true;++update_attempt_;
  try{
    Config candidate=Config::parse(text);
    if(candidate.geometry.mode=="calibrated_3d"&&input_size_.area()!=0)calibrated_config(candidate,input_size_);
    // Policy may change without changing the effective maps (for example,
    // equivalent K/reference scaling). Validate it even when maps are reused.
    if(candidate.geometry.undistort&&input_size_.area()!=0){
      const auto& g=candidate.geometry;
      const double sx=double(input_size_.width)/g.calibration_width;
      const double sy=double(input_size_.height)/g.calibration_height;
      if(input_size_!=cv::Size(g.calibration_width,g.calibration_height)&&
         (!g.allow_uniform_resize||std::abs(sx-sy)>1e-9))
        throw std::runtime_error("calibration/video resolution mismatch");
    }
    if(candidate.input!=current_.input||candidate.timestamp_file!=current_.timestamp_file||
       candidate.prefetch_frames!=current_.prefetch_frames)throw std::runtime_error("input change requires restart");
    if(candidate.geometry.mode!=current_.geometry.mode)throw std::runtime_error("estimator mode change requires restart");
    if(candidate.detector_runtime_signature!=current_.detector_runtime_signature)
      throw std::runtime_error("detector backend/model settings require restart; previous configuration retained");
    const bool reset=candidate.estimation_signature!=current_.estimation_signature;
    // Build and validate maps before committing either configuration or geometry.
    const bool remap_changed=candidate.preprocessing_signature!=current_.preprocessing_signature;
    std::optional<ImagePreprocessor> prepared;
    if(input_size_.area()!=0&&remap_changed)prepared.emplace(candidate.geometry,input_size_);
    if(prepared)preprocessor_=std::move(*prepared);
    current_=std::move(candidate);message=reset?"config applied; incompatible tracks reset":"config applied live";
    last_rejected_=false;last_rejection_.clear();
    return reset?ReloadResult::ResetRequired:ReloadResult::LiveUpdated;
  }catch(const std::exception& e){message=e.what();last_rejected_=true;last_rejection_=message;return ReloadResult::Rejected;}
}
ReloadResult ConfigRuntime::poll(std::string& message){
  std::ifstream file(path_);
  if(!file){message="config unavailable; retaining previous snapshot";return ReloadResult::Rejected;}
  std::ostringstream data;data<<file.rdbuf();return apply_text(data.str(),message);
}
VideoSource::VideoSource(const std::string& path,const std::string& timestamps,int prefetch) : capture_(path),capacity_(prefetch) {
  if(prefetch<0||prefetch>8)throw std::runtime_error("prefetch_frames must be 0..8");
  if (!capture_.isOpened()) throw std::runtime_error("cannot open video: " + path);
  fps_ = capture_.get(cv::CAP_PROP_FPS);
  if (!std::isfinite(fps_) || fps_ <= 0 || fps_ > 1000)
    throw std::runtime_error("invalid video frame rate");
  const double count=capture_.get(cv::CAP_PROP_FRAME_COUNT);
  if(std::isfinite(count)&&count>0&&count<double(std::numeric_limits<long>::max())&&
     std::abs(count-std::round(count))<1e-6)reported_frames_=std::lround(count);
  if(!timestamps.empty()){
    std::ifstream input(timestamps);std::string line;
    if(!input||!std::getline(input,line)||line!="frame,time_seconds")
      throw std::runtime_error("timestamp sidecar requires frame,time_seconds header");
    while(std::getline(input,line)){
      std::istringstream row(line);long index;double time;char comma;
      if(!(row>>index>>comma>>time)||comma!=','||(row>>std::ws&&!row.eof())||
         index!=long(stamps_.size())||!std::isfinite(time)||time<0||
         (!stamps_.empty()&&time<=stamps_.back()))
        throw std::runtime_error("invalid/nonmonotonic timestamp sidecar row");
      stamps_.push_back(time);
    }
    if(stamps_.empty()||(reported_frames_>=0&&long(stamps_.size())!=reported_frames_))
      throw std::runtime_error("timestamp/video frame count mismatch");
  }
  if(capacity_)worker_=std::thread(&VideoSource::produce,this);
}
VideoSource::~VideoSource(){
  {std::lock_guard<std::mutex> lock(mutex_);stopped_=true;}
  space_.notify_all();ready_.notify_all();if(worker_.joinable())worker_.join();
}
size_t VideoSource::queue_high_water()const{std::lock_guard<std::mutex> lock(mutex_);return high_water_;}
long VideoSource::forecast_frame(long frame,double horizon)const{
  if(frame<0||!std::isfinite(horizon)||horizon<=0)throw std::runtime_error("invalid forecast timestamp request");
  if(stamps_.empty()){
    const double step=horizon*fps_;
    if(step<1||step>double(std::numeric_limits<long>::max()-frame)||std::abs(step-std::round(step))>1e-6)
      throw std::runtime_error("forecast horizon not aligned with CFR timestamps");
    return frame+std::lround(step);
  }
  if(size_t(frame)>=stamps_.size())throw std::runtime_error("forecast source frame outside timestamps");
  const double target=stamps_[size_t(frame)]+horizon;
  if(target>stamps_.back()+1e-9)return -2;
  const auto it=std::lower_bound(stamps_.begin()+frame+1,stamps_.end(),target-1e-9);
  return it!=stamps_.end()&&std::abs(*it-target)<=1e-9?long(it-stamps_.begin()):-1;
}
double VideoSource::frame_interval(long frame)const{
  return !stamps_.empty()&&frame>=0&&size_t(frame+1)<stamps_.size()?
    stamps_[size_t(frame+1)]-stamps_[size_t(frame)]:1/fps_;
}
void VideoSource::produce(){
  try{while(true){
    {std::unique_lock<std::mutex> lock(mutex_);space_.wait(lock,[&]{return stopped_||queue_.size()<size_t(capacity_);});if(stopped_)break;}
    Frame frame;if(!decode(frame))break;
    {std::lock_guard<std::mutex> lock(mutex_);if(stopped_)break;queue_.push_back(std::move(frame));high_water_=std::max(high_water_,queue_.size());}
    ready_.notify_one();
  }}catch(...){std::lock_guard<std::mutex> lock(mutex_);error_=std::current_exception();}
  {std::lock_guard<std::mutex> lock(mutex_);done_=true;}ready_.notify_all();
}
bool VideoSource::next(Frame& frame){
  if(!capacity_)return decode(frame);
  std::unique_lock<std::mutex> lock(mutex_);ready_.wait(lock,[&]{return !queue_.empty()||done_;});
  if(queue_.empty()){frame.image.release();if(error_)std::rethrow_exception(error_);return false;}
  frame=std::move(queue_.front());queue_.pop_front();lock.unlock();space_.notify_one();return true;
}
bool VideoSource::decode(Frame& frame) {
  if(ended_){frame.image.release();return false;}
  if (!capture_.read(frame.image)) {
    if(reported_frames_>=0&&index_!=reported_frames_)
      throw std::runtime_error("video decode count mismatch: decoded="+std::to_string(index_)+
                               " declared="+std::to_string(reported_frames_)+
                               "; cannot confirm complete input (decode failure or inaccurate metadata)");
    if(!stamps_.empty()&&index_!=long(stamps_.size()))throw std::runtime_error("timestamp/video EOF mismatch");
    frame.image.release();ended_=true;return false;
  }
  if(frame.image.empty())throw std::runtime_error("decoder returned an empty video frame");
  frame.index = index_++;
  if(!stamps_.empty()&&frame.index>=long(stamps_.size()))throw std::runtime_error("missing frame timestamp");
  frame.time_seconds = stamps_.empty()?frame.index / fps_:stamps_[size_t(frame.index)];
  return true;
}
}  // namespace outpost
