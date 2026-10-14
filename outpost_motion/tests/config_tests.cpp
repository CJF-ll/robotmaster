#include "core.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>
void require(bool v,const char* text){if(!v)throw std::runtime_error(text);}
int main(int argc,char**argv){
  if(argc!=2)return 2;
  outpost::ConfigRuntime runtime(argv[1]);std::string message;
  auto root=YAML::LoadFile(argv[1]);
  {outpost::ConfigRuntime strict(argv[1]);const auto original=strict.current().source_text;
    auto reject=[&](const std::string& text,const std::string& key){
      require(strict.apply_text(text,message)==outpost::ReloadResult::Rejected,"unknown/duplicate config committed");
      require(message.find(key)!=std::string::npos,"config error missing field path");
      require(strict.current().source_text==original,"schema rejection changed active config");
    };
    auto cfg=YAML::Load(original);cfg["tracking"]["measurment_noise"]=9;
    reject(YAML::Dump(cfg),"tracking.measurment_noise");
    cfg=YAML::Load(original);cfg["detector"]["neural"]["score_thresold"]=.5;
    reject(YAML::Dump(cfg),"detector.neural.score_thresold");
    cfg=YAML::Load(original);cfg["geometry"]["rotation_radius_cm"]=30;
    reject(YAML::Dump(cfg),"geometry.rotation_radius_cm");
    cfg=YAML::Load(original);cfg["camera"]["enable"]=true;
    reject(YAML::Dump(cfg),"camera.enable");
    reject(original+"\nschema_version: 1\n","schema_version");
    cfg=YAML::Load(original);cfg.remove("prediction");
    reject(YAML::Dump(cfg)+"\nprediction:\n  horizon_seconds: 0.1\n  horizon_seconds: 0.2\n","prediction.horizon_seconds");
    cfg=YAML::Load(original);cfg["tracking"]=YAML::Load("[1,2]");
    reject(YAML::Dump(cfg),"tracking");
    require(strict.apply_text(original,message)==outpost::ReloadResult::LiveUpdated,"valid configuration did not recover");
  }
  {outpost::ConfigRuntime audit(argv[1]);const auto initial=audit.current().source_text;
    require(!initial.empty()&&outpost::Config::parse(initial).input==audit.current().input,"startup source snapshot missing");
    auto cfg=YAML::Load(initial);cfg["prediction"]["horizon_seconds"]=.15;
    const auto accepted=YAML::Dump(cfg);
    require(audit.apply_text(accepted,message)==outpost::ReloadResult::LiveUpdated&&audit.current().source_text==accepted,"accepted source not retained");
    const auto version=audit.update_attempt();
    require(audit.apply_text(accepted,message)==outpost::ReloadResult::Unchanged&&audit.update_attempt()==version,"unchanged text created new attempt");
    require(audit.apply_text("",message)==outpost::ReloadResult::Rejected&&audit.current().source_text==accepted&&audit.update_attempt()==version+1,"rejected source replaced accepted provenance");
    audit.apply_text("",message);require(audit.update_attempt()==version+1,"cached rejection advanced attempt");
    audit.apply_text(" ",message);require(audit.update_attempt()==version+2,"different invalid edits conflated");
  }
  {outpost::ConfigRuntime empty(argv[1]);
    require(empty.apply_text("",message)==outpost::ReloadResult::Rejected,"initial empty edit treated as unchanged");
    require(empty.apply_text("",message)==outpost::ReloadResult::Rejected&&!message.empty(),"repeated empty edit lost rejection");
  }
  {outpost::ConfigRuntime maps(argv[1]);auto cfg=YAML::LoadFile(argv[1]);
    // Synthetic camera calibration tests map lifecycle, not this video's K/D.
    cfg["geometry"]["undistort"]=true;cfg["geometry"]["calibration_width"]=64;cfg["geometry"]["calibration_height"]=48;
    cfg["geometry"]["camera_matrix"]=std::vector<double>{100,0,32,0,100,24,0,0,1};
    cfg["geometry"]["distortion"]=std::vector<double>{0,0,0,0,0};
    require(maps.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::ResetRequired,"calibration not applied");
    maps.bind_input_size({64,48});const auto* first=maps.preprocessor().camera_matrix().data;
    require(first&&maps.preprocessor().rectified(),"synthetic maps missing");
    cfg["prediction"]["horizon_seconds"]=.15;
    require(maps.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::LiveUpdated&&maps.preprocessor().camera_matrix().data==first,"horizon unnecessarily rebuilt maps");
    cfg["tracking"]["acceleration_noise"]=30000;
    require(maps.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::ResetRequired&&maps.preprocessor().camera_matrix().data==first,"motion reset unnecessarily rebuilt maps");
    cfg["geometry"]["armor_width_m"]=.135;
    require(maps.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::LiveUpdated&&maps.preprocessor().camera_matrix().data==first,"unused board dimensions reset 2D state or camera maps");
    const auto good=YAML::Dump(cfg);cfg["geometry"]["calibration_width"]=65;const auto bad=YAML::Dump(cfg);
    require(maps.apply_text(bad,message)==outpost::ReloadResult::Rejected,"mismatched calibration committed");
    const auto rejection=message;
    require(maps.apply_text(bad,message)==outpost::ReloadResult::Rejected&&message==rejection,"cached bad configuration reported unchanged");
    require(maps.preprocessor().camera_matrix().data==first&&maps.current().geometry.calibration_width==64,"failed remap replaced valid snapshot");
    require(maps.apply_text(good,message)==outpost::ReloadResult::LiveUpdated,"corrected configuration not recovered");
    require(maps.apply_text(good,message)==outpost::ReloadResult::Unchanged&&message.empty(),"unchanged valid config retained stale message");
    cfg=YAML::Load(good);cfg["geometry"]["camera_matrix"]=std::vector<double>{110,0,32,0,110,24,0,0,1};
    require(maps.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::ResetRequired&&maps.preprocessor().camera_matrix().data!=first&&maps.preprocessor().camera_matrix().at<double>(0,0)==110,"changed camera did not rebuild maps");
  }
  {outpost::ConfigRuntime dimensions(argv[1]);auto cfg=YAML::LoadFile(argv[1]);
    cfg["geometry"]["armor_height_m"]=.126;
    cfg["geometry"]["lightbar_length_m"]=.056;
    cfg["geometry"]["lightbar_center_distance_m"]=.135;
    require(dimensions.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::LiveUpdated,"inactive physical dimensions reset image-only estimator");
    require(*dimensions.current().geometry.armor_height_m==.126&&*dimensions.current().geometry.lightbar_length_m==.056,"board and lightbar dimensions conflated");
    cfg["geometry"]["lightbar_length_m"]=-1;
    require(dimensions.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::Rejected,"negative lightbar dimension accepted");
  }
  {outpost::ConfigRuntime backend(argv[1]);auto cfg=YAML::LoadFile(argv[1]);
    cfg["detector"]["neural"]["score_threshold"]=.8;
    require(backend.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::Rejected&&backend.current().detector.neural_score==.7,"restart-only setting committed live");
    cfg["detector"]["neural"]["score_threshold"]=-1;
    bool bad=false;try{outpost::Config::parse(YAML::Dump(cfg));}catch(...){bad=true;}require(bad,"invalid neural score accepted");
  }
  {outpost::ConfigRuntime threads(argv[1]);auto cfg=YAML::LoadFile(argv[1]);
    require(threads.current().detector.neural_inference_threads==0,"thread default changed");
    for(int value:{0,1,2,256}){cfg["detector"]["neural"]["inference_threads"]=value;
      require(outpost::Config::parse(YAML::Dump(cfg)).detector.neural_inference_threads==value,"valid thread setting rejected");}
    for(int value:{-1,257}){cfg["detector"]["neural"]["inference_threads"]=value;
      bool bad=false;try{outpost::Config::parse(YAML::Dump(cfg));}catch(...){bad=true;}
      require(bad,"invalid thread setting accepted");}
    cfg["detector"]["neural"]["inference_threads"]=2;
    require(threads.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::Rejected&&
      threads.current().detector.neural_inference_threads==0,"thread setting changed without restart");
  }
  {outpost::ConfigRuntime guarded(argv[1]);auto cfg=YAML::LoadFile(argv[1]);
    cfg["tracking"]["one_sided_guard"]["enabled"]=true;
    require(guarded.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::ResetRequired,"guard toggle did not reset");
    cfg["tracking"]["one_sided_guard"]["shrink_ratio"]=1.1;
    require(guarded.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::Rejected&&guarded.current().tracker.one_sided_shrink_ratio==.7,"invalid guard changed snapshot");
  }
  {outpost::ConfigRuntime vis_runtime(argv[1]);auto cfg=YAML::LoadFile(argv[1]);
    cfg["visibility"]["min_axial_overlap"]=.8;
    require(vis_runtime.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::LiveUpdated,
            "visibility overlap did not update live");
    cfg["visibility"]["min_axial_overlap"]=1.1;
    require(vis_runtime.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::Rejected&&
            vis_runtime.current().visibility.min_axial_overlap==.8,"invalid overlap corrupted config");
    cfg["visibility"]["min_axial_overlap"]=-.1;
    require(vis_runtime.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::Rejected,
            "negative overlap accepted");
  }
  {outpost::ConfigRuntime health_runtime(argv[1]);auto cfg=YAML::LoadFile(argv[1]);
    cfg["tracking"]["health"]["min_samples"]=10;
    require(health_runtime.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::ResetRequired,
            "health window config not applied");
    cfg["tracking"]["health"]["failure_fraction"]=1.1;
    require(health_runtime.apply_text(YAML::Dump(cfg),message)==outpost::ReloadResult::Rejected&&
            health_runtime.current().tracker.health.failure_fraction<=1,"bad health threshold committed");
  }
  outpost::ConfigRuntime quality_runtime(argv[1]);auto quality_root=YAML::LoadFile(argv[1]);
  quality_root["tracking"]["low_resolution_noise_scale"]=9;
  require(quality_runtime.apply_text(YAML::Dump(quality_root),message)==outpost::ReloadResult::ResetRequired,
          "quality noise change not reset");
  quality_root["tracking"]["low_resolution_noise_scale"]=.5;
  require(quality_runtime.apply_text(YAML::Dump(quality_root),message)==outpost::ReloadResult::Rejected&&
          quality_runtime.current().tracker.low_resolution_noise_scale==9,"invalid quality noise changed snapshot");
  outpost::ConfigRuntime border_runtime(argv[1]);
  outpost::ConfigRuntime thin_runtime(argv[1]);
  auto thin_root=YAML::LoadFile(argv[1]);
  thin_root["detector"]["thin_min_pixels"]=10;
  require(thin_runtime.apply_text(YAML::Dump(thin_root),message)==outpost::ReloadResult::ResetRequired,
          "thin geometry change not reset");
  thin_root["detector"]["thin_min_pixels"]=2;
  require(thin_runtime.apply_text(YAML::Dump(thin_root),message)==outpost::ReloadResult::Rejected&&
          thin_runtime.current().detector.thin_min_pixels==10,"invalid thin gate changed active config");
  thin_root["detector"]["thin_min_pixels"]=10;
  thin_root["detector"]["thin_vertical_enabled"]=false;
  require(thin_runtime.apply_text(YAML::Dump(thin_root),message)==outpost::ReloadResult::ResetRequired,
          "thin mode change did not reset incompatible geometry");
  root["detector"]["border_margin_px"]=2;
  require(border_runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::ResetRequired,"border gate change not reset");
  root["detector"]["border_margin_px"]=-1;
  require(border_runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::Rejected&&
          border_runtime.current().detector.border_margin_px==2,"invalid border update not atomic");
  root["detector"]["border_margin_px"]=0;
  require(border_runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::ResetRequired,"border gate restore not reset");
  root["tracking"]["association_length_ratio"]=2.0;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::LiveUpdated,"equivalent YAML reset tracking");
  root["prediction"]["horizon_seconds"]=.15;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::LiveUpdated,"horizon not live");
  root["detector"]["brightness"]=140;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::LiveUpdated,"threshold reset tracks");
  root["detector"]["end_margin_min_px"]=2;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::ResetRequired,"enclosure change did not reset incompatible shape");
  root["detector"]["side_margin_min_px"]=-1;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::Rejected,"negative enclosure margin accepted");
  require(runtime.current().detector.end_margin_min_px==2&&runtime.current().detector.side_margin_min_px>=0,
          "invalid enclosure snapshot corrupted active config");
  root["detector"]["side_margin_min_px"]=.75;
  root["detector"]["pairing_ambiguity_margin"]=.1;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::ResetRequired,"pairing rule change preserved incompatible state");
  root["detector"]["pairing_ambiguity_margin"]=-.1;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::Rejected,"negative pairing ambiguity margin accepted");
  require(runtime.current().detector.pairing_ambiguity_margin==.1,"invalid pairing config changed active snapshot");
  root["detector"]["pairing_ambiguity_margin"]=.1;
  const double before=runtime.current().prediction_seconds;
  root["prediction"]["horizon_seconds"]=-1;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::Rejected,"invalid snapshot committed");
  require(runtime.current().prediction_seconds==before,"invalid config replaced good snapshot");
  root["prediction"]["horizon_seconds"]=.15;
  root["tracking"]["acceleration_noise"]=30000;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::ResetRequired,"model change preserved incompatible state");
  root["tracking"]["association_miss_cost"]=16;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::ResetRequired,"association policy change not reset");
  root["tracking"]["association_miss_cost"]=-1;
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::Rejected,"negative unmatched cost accepted");
  require(runtime.current().tracker.association_miss_cost==16,"invalid association config replaced snapshot");
  root["tracking"]["association_miss_cost"]=16;
  root["geometry"]["slot_height_offsets_m"]=std::vector<double>{0,.03,.06};
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::LiveUpdated,"inactive physical slot heights reset image-only estimator");
  require(runtime.current().geometry.slot_height_offsets_m.size()==3,"geometry not parsed");
  root["geometry"]["camera_matrix"]=std::vector<double>{1,2};
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::Rejected,"malformed K accepted");
  root["geometry"]["camera_matrix"]=YAML::Null;
  root["input"]["path"]="another.mp4";
  require(runtime.apply_text(YAML::Dump(root),message)==outpost::ReloadResult::Rejected,"input silently changed midstream");
  std::cout<<"atomic config reload and invalid rollback passed\n";
}
