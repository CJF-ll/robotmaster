#include "core.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>
using namespace outpost;
void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(int argc,char** argv){try{
  if(argc!=2)return 2;const auto original=YAML::LoadFile(argv[1]);
  auto rejected=[](const YAML::Node& node){try{Config::parse(YAML::Dump(node));return false;}catch(...){return true;}};
  auto clone=[&](){return YAML::Clone(original);};std::string message;
  {ConfigRuntime input(argv[1]);auto changed=clone();changed["input"]["prefetch_frames"]=2;
    check(input.apply_text(YAML::Dump(changed),message)==ReloadResult::Rejected,"queue policy changed without restart");
    changed=clone();changed["input"]["timestamp_file"]="/missing/timestamps.csv";
    check(input.apply_text(YAML::Dump(changed),message)==ReloadResult::Rejected,"time source changed without restart");
    changed=clone();changed["input"]["prefetch_frames"]=9;check(rejected(changed),"unbounded prefetch accepted");
    changed=clone();changed["tracking"]["shape_guard"]["enabled"]=true;
    check(input.apply_text(YAML::Dump(changed),message)==ReloadResult::ResetRequired,"shape gate enabled without state reset");
    changed["tracking"]["shape_guard"]["history_samples"]=1;check(rejected(changed),"two-frame shape evidence accepted");
    changed=clone();changed["rigid_estimator"]["association_nis_scale"]=0;check(rejected(changed),"invalid normalized innovation scale accepted");}
  auto c=clone();c["geometry"]["mode"]="calibrated_3d";
  check(rejected(c),"unknown calibration permitted real 3D execution");
  c=clone();c["geometry"]["camera_from_target_rotation"]=std::vector<double>{1,0,0,0,1,0,0,0,-1};
  check(rejected(c),"reflection accepted as target axis rotation");
  c=clone();c["tracking"]["measurement_quality"]["enabled"]=true;
  ConfigRuntime quality(argv[1]);check(quality.apply_text(YAML::Dump(c),message)==ReloadResult::ResetRequired,"quality update failed to reset");
  c["tracking"]["measurement_quality"]["max_variance_scale"]=.5;
  check(quality.apply_text(YAML::Dump(c),message)==ReloadResult::Rejected,"bad quality update committed");
  c=clone();c["tracking"]["clean_recovery"]["enabled"]=true;
  ConfigRuntime recovery(argv[1]);check(recovery.apply_text(YAML::Dump(c),message)==ReloadResult::ResetRequired,"recovery update failed to reset");
  c["tracking"]["clean_recovery"]["max_gap_seconds"]=1;
  check(rejected(c),"recovery lifetime exceeds coast deadline");
  c=clone();c["detector"]["neural"]["roi_refinement"]["enabled"]=true;
  ConfigRuntime roi(argv[1]);check(roi.apply_text(YAML::Dump(c),message)==ReloadResult::Rejected,"ROI activation changed without restart");
  c["detector"]["neural"]["roi_refinement"]["brightness"]=256;check(rejected(c),"bad ROI threshold accepted");
  c=clone();auto g=c["geometry"];g["mode"]="calibrated_3d";
  // Synthetic calibration used only in this test, never in the supplied video.
  g["camera_matrix"]=std::vector<double>{500,0,320,0,500,240,0,0,1};
  g["distortion"]=std::vector<double>{.01,0,0,0,0};
  g["calibration_width"]=640;g["calibration_height"]=480;
  g["camera_from_target_rotation"]=std::vector<double>{1,0,0,0,0,-1,0,1,0};
  g["lightbar_length_m"]=.056;g["lightbar_center_distance_m"]=.135;
  g["rotation_radius_m"]=.25;g["armor_pitch_deg"]=0;
  g["slot_height_offsets_m"]=std::vector<double>{0,.03,.07};
  auto config=Config::parse(YAML::Dump(c));const auto raw=calibrated_config(config,{640,480});
  const auto signature=config.estimation_signature;
  auto inactive=YAML::Clone(c);
  inactive["tracking"]["measurement_noise"]=12;
  inactive["target_slots"]["count"]=4;
  inactive["target_slots"]["sequence_transitions"]=8;
  inactive["geometry"]["armor_width_m"]=.20;
  check(Config::parse(YAML::Dump(inactive)).estimation_signature==signature,
        "inactive image tracking/board settings reset calibrated estimator");
  auto active=YAML::Clone(c);active["geometry"]["lightbar_length_m"]=.07;
  check(Config::parse(YAML::Dump(active)).estimation_signature!=signature,"active rigid geometry missing from lifecycle");
  active=YAML::Clone(c);active["rigid_estimator"]["pixel_noise_std_px"]=1;
  check(Config::parse(YAML::Dump(active)).estimation_signature!=signature,"uncertainty policy missing from lifecycle");
  auto equivalent=YAML::Clone(c);equivalent["geometry"]["calibration_width"]=1280;
  equivalent["geometry"]["calibration_height"]=960;
  equivalent["geometry"]["camera_matrix"]=std::vector<double>{1000,0,640,0,1000,480,0,0,1};
  check(Config::parse(YAML::Dump(equivalent)).estimation_signature==signature,"equivalent normalized K reset state");
  auto image=clone();const auto image_signature=Config::parse(YAML::Dump(image)).estimation_signature;
  image["geometry"]["camera_matrix"]=std::vector<double>{500,0,320,0,500,240,0,0,1};
  image["geometry"]["distortion"]=std::vector<double>{.01,0,0,0,0};
  image["geometry"]["calibration_width"]=640;image["geometry"]["calibration_height"]=480;
  image["rigid_estimator"]["pixel_noise_std_px"]=1;
  check(Config::parse(YAML::Dump(image)).estimation_signature==image_signature,"inactive raw-image calibration reset state");
  check(raw.distortion.at<double>(0)==.01&&raw.camera_matrix.at<double>(0,0)==500,"raw calibration changed");
  config.geometry.undistort=true;const auto rectified=calibrated_config(config,{640,480});
  check(cv::norm(rectified.distortion)==0,"distortion applied twice after remapping");
  bool mismatch=false;try{calibrated_config(config,{320,240});}catch(...){mismatch=true;}
  check(mismatch,"unapproved calibration resize accepted");
  config.geometry.allow_uniform_resize=true;
  check(calibrated_config(config,{320,240}).camera_matrix.at<double>(0,0)==250,"effective intrinsics not resized");
  ConfigRuntime mode(argv[1]);check(mode.apply_text(YAML::Dump(c),message)==ReloadResult::Rejected,"runtime estimator mode silently changed");
  c["rigid_estimator"]["max_phase_error_rad"]=4;check(rejected(c),"invalid angular association gate accepted");
  std::cout<<"optimization configuration, calibration completeness and coordinate contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
