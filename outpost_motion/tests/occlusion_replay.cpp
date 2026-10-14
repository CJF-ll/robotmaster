#include "core.hpp"
#include "estimator.hpp"
#include "detection_backend.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace outpost;
struct CachedFrame {
  long index;double time;std::vector<Detection> observations;
  std::uint64_t reference_id=0;int reference_index=0;
};
struct Clip {size_t first;int length,warmup,recovery,lead;std::uint64_t reference_id;};
struct Score {
  long opportunities=0,model_issued=0,observed_issued=0,id_kept=0,cases=0,recovered=0;
  long id_present=0,wrong_reference_matches=0,missing_recovery_observations=0,mature_before_mask=0;
  long moving_opportunities=0,moving_observed_issued=0;
  double wait_sum=0,wait_frames_sum=0;std::vector<double> errors,added,moving_errors,added_moving;
};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
// This function receives only the current or earlier detector output on every
// step. Reference selection and future residual scoring are outside this path.
std::vector<std::vector<TrackOutput>> replay(const std::vector<CachedFrame>& frames,
    const Clip& clip,int missing,TrackerConfig config,double horizon){
  Tracker tracker(config);std::vector<std::vector<TrackOutput>> result;
  for(int i=0;i<clip.length;++i){const auto& f=frames.at(clip.first+size_t(i));
    const bool masked=i>=clip.warmup&&i<clip.warmup+missing;
    result.push_back(tracker.step(f.time,masked?std::vector<Detection>{}:f.observations,horizon));
  }
  return result;
}
std::vector<Clip> clips(const std::vector<CachedFrame>& frames,int warmup,int recovery,int lead){
  // Selection uses the unmodified recording's continuous local identity only,
  // never the fast policy's errors, success, NIS or future acceptance rate.
  const int length=warmup+3+recovery+lead;std::vector<Clip> result;
  for(size_t i=0;i<frames.size();){
    if(!frames[i].reference_id){++i;continue;}
    const size_t begin=i;const auto id=frames[i].reference_id;
    while(i<frames.size()&&frames[i].reference_id==id)++i;
    for(size_t first=begin;first+size_t(length)<=i;first+=size_t(length))
      result.push_back({first,length,warmup,recovery,lead,id});
  }
  return result;
}
const TrackOutput* by_id(const std::vector<TrackOutput>& tracks,std::uint64_t id){
  for(const auto& t:tracks)if(t.id==id)return &t;
  return nullptr;
}
void identical_state(const std::vector<TrackOutput>& a,const std::vector<TrackOutput>& b){
  require(a.size()==b.size(),"recovery policy changed track count");
  for(size_t i=0;i<a.size();++i)require(a[i].id==b[i].id&&a[i].measurement_used==b[i].measurement_used&&
    a[i].detection_index==b[i].detection_index&&a[i].current==b[i].current&&a[i].future==b[i].future&&
    a[i].filtered_center==b[i].filtered_center&&a[i].velocity==b[i].velocity&&
    a[i].velocity_variance==b[i].velocity_variance&&a[i].future_position_variance==b[i].future_position_variance,
    "recovery admission modified association, filter or silhouette");
}
std::string stats(std::vector<double> errors){
  if(errors.empty())return "n=0";
  std::sort(errors.begin(),errors.end());double sum=0;for(double e:errors)sum+=e;
  std::ostringstream s;s<<std::fixed<<std::setprecision(5)<<"n="<<errors.size()<<" mean="<<sum/errors.size()
    <<" p95="<<errors[size_t(std::ceil(.95*errors.size()))-1]<<" max="<<errors.back();return s.str();
}
Detection synthetic(float x){Detection d;d.center={x,50};
  d.corners={cv::Point2f(x-10,44),{x+10,44},{x+10,56},{x-10,56}};d.measurement_points=d.corners;return d;}
void self_test(){
  std::vector<CachedFrame> frames;for(int i=0;i<60;++i)frames.push_back({i,i/30.,{synthetic(100+i)},1});
  const auto selected=clips(frames,12,8,3);
  require(selected.size()==2&&selected[0].first==0&&selected[1].first==26,"fixed nonoverlapping clip selection failed");
  TrackerConfig base,fast;fast.clean_recovery_enabled=true;
  for(int missing:{1,2,3}){const auto a=replay(frames,selected[0],missing,base,.1),b=replay(frames,selected[0],missing,fast,.1);
    for(size_t i=0;i<a.size();++i)identical_state(a[i],b[i]);
    const int second=selected[0].warmup+missing+1;
    require(b[size_t(second)].at(0).future_valid==(missing<=2),"replay recovery boundary incorrect");
    for(int i=12;i<12+missing;++i)require(!b[size_t(i)].at(0).measurement_used,"masked detection leaked into observation");
  }
  auto changed=frames;for(size_t i=20;i<changed.size();++i)changed[i].observations={synthetic(500)};
  const auto before=replay(frames,selected[0],1,fast,.1),after=replay(changed,selected[0],1,fast,.1);
  for(int i=0;i<20;++i){identical_state(before[size_t(i)],after[size_t(i)]);
    require(before[size_t(i)].empty()||before[size_t(i)][0].future_valid==after[size_t(i)][0].future_valid,
            "future observation changed earlier forecast admission");}
  std::cout<<"fixed clip selection, masking, matched-state comparison and causal replay passed\n";
}
int main(int argc,char** argv){try{
  if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
  if(argc!=3&&argc!=5){std::cerr<<"usage: occlusion_replay config.yaml new_output_directory [warmup_frames recovery_frames]\n";return 2;}
  auto config=Config::load(argv[1]);
  require_estimator_mode(config,"image_2d","occlusion_replay");
  require(config.timestamp_file.empty(),"frame-count occlusion replay requires CFR; timestamp sidecars are not supported by this tool");
  VideoSource source(config);
  auto count=[](const char* value){size_t used=0;const int n=std::stoi(value,&used);
    require(used==std::string(value).size()&&n>0&&n<=10000,"invalid replay frame count");return n;};
  const int warmup=argc==5?count(argv[3]):std::max(8,config.tracker.prediction_hits+2);
  const int recovery=argc==5?count(argv[4]):std::max(7,config.tracker.prediction_hits+2);
  require(warmup>=config.tracker.prediction_hits&&recovery>=config.tracker.prediction_hits,
          "replay must preserve full maturity and recovery windows");
  const double lead_float=config.prediction_seconds*source.fps();const int lead=int(std::lround(lead_float));
  require(lead>=1&&std::abs(lead_float-lead)<1e-6,"horizon must be a positive integer number of video frames");
  require(std::filesystem::create_directory(argv[2]),"output directory already exists or cannot be created");
  const std::filesystem::path output(argv[2]);std::ofstream snapshot(output/"config_0.yaml");
  snapshot<<config.source_text;snapshot.close();require(bool(snapshot),"cannot write config snapshot");
  auto detector=make_detector(config.detector.backend,config.detector);auto original=config.tracker;
  original.clean_recovery_enabled=false;original.measurement_quality_enabled=false;Tracker reference(original);
  std::vector<CachedFrame> frames;Frame frame;std::optional<ImagePreprocessor> preprocessing;
  std::ofstream measurements(output/"reference_observations.csv");
  measurements<<std::setprecision(17)<<"frame,time,detection_index,reference_local_id,cx,cy,low_resolution";
  for(int i=0;i<4;++i)measurements<<",lamp"<<i<<"_x,lamp"<<i<<"_y";
  measurements<<'\n';
  while(source.next(frame)){
    if(!preprocessing)preprocessing.emplace(config.geometry,frame.image.size());
    const auto detected=detector->detect(preprocessing->process(frame.image));
    const auto tracks=reference.step(frame.time_seconds,detected.observations,config.prediction_seconds);
    std::uint64_t id=0;int reference_index=0;
    // First accepted local ID is deterministic (creation order). Other real
    // candidates and coasting tracks are retained in every replay input.
    for(const auto& track:tracks)if(track.measurement_used){id=track.id;reference_index=track.detection_index;break;}
    frames.push_back({frame.index,frame.time_seconds,detected.observations,id,reference_index});
    for(size_t j=0;j<detected.observations.size();++j){const auto& d=detected.observations[j];
      measurements<<frame.index<<','<<frame.time_seconds<<','<<j<<','<<(int(j)==reference_index?id:0)<<','<<d.center.x<<','<<d.center.y<<','<<d.low_resolution;
      for(auto p:d.measurement_points)measurements<<','<<p.x<<','<<p.y;
      measurements<<'\n';}
  }
  measurements.close();require(bool(measurements),"reference observation audit failed");
  const auto selected=clips(frames,warmup,recovery,lead);require(!selected.empty(),"no sufficiently long continuously observed local-ID clips");
  auto candidate=original;candidate.clean_recovery_enabled=true;
  require(candidate.clean_recovery_hits>=2&&candidate.clean_recovery_hits<=candidate.prediction_hits&&
    std::isfinite(candidate.clean_recovery_max_gap_seconds)&&candidate.clean_recovery_max_gap_seconds>0&&
    candidate.clean_recovery_max_gap_seconds<=candidate.coast_seconds&&
    std::isfinite(candidate.clean_recovery_max_nis)&&candidate.clean_recovery_max_nis>0&&
    candidate.clean_recovery_max_nis<=candidate.innovation_gate,"invalid experimental clean recovery parameters");
  std::ofstream manifest(output/"cases.csv"),rows(output/"forecasts.csv"),case_results(output/"case_results.csv");
  manifest<<"case,first_frame,last_frame,reference_local_id,warmup_frames,drop_start_frame,missing_frames,recovery_frames,horizon_frames\n";
  rows<<std::setprecision(17)<<"case,missing_frames,mode,frame,phase,track_present,id,measurement_used,future_valid,observed_future_eligible,forecast_blocks,clean_recovery_used,target_frame,reference_x,reference_y,predicted_x,predicted_y,error_px\n";
  case_results<<"case,missing_frames,mode,replay_target_id,mature_before_mask,id_present_through_window,reference_association_complete,wrong_reference_matches,missing_recovery_observations,recovery_wait_accepted_frames,recovery_wait_frames,model_issued,observed_issued,opportunities\n";
  Score totals[3][2];int case_id=0;
  for(const auto& clip:selected)for(int missing:{1,2,3}){
    ++case_id;manifest<<case_id<<','<<frames[clip.first].index<<','<<frames[clip.first+size_t(clip.length-1)].index<<','
      <<clip.reference_id<<','<<clip.warmup<<','<<frames[clip.first+size_t(clip.warmup)].index<<','<<missing<<','<<clip.recovery<<','<<clip.lead<<'\n';
    // Both policies receive the same frozen detections/mask. Scoring starts only
    // after both complete; future reference observations cannot enter filtering.
    const auto baseline=replay(frames,clip,missing,original,config.prediction_seconds);
    const auto optimized=replay(frames,clip,missing,candidate,config.prediction_seconds);
    for(size_t i=0;i<baseline.size();++i)identical_state(baseline[i],optimized[i]);
    std::uint64_t target_id=0;
    for(const auto& t:baseline[size_t(clip.warmup-1)])
      if(t.measurement_used&&t.detection_index==frames[clip.first+size_t(clip.warmup-1)].reference_index)target_id=t.id;
    const auto* pre_mask=by_id(baseline[size_t(clip.warmup-1)],target_id);
    const bool mature_before_mask=pre_mask&&!(pre_mask->forecast.blocks&ForecastHistory);
    // Choosing which output to audit uses only the last pre-mask accepted
    // measurement. It does not relabel the tracker or repair an eventual swap.
    const int finish=clip.warmup+missing+clip.recovery;
    for(int mode=0;mode<2;++mode){const auto& result=mode?optimized:baseline;auto& total=totals[missing-1][mode];
      const char* name=mode?"clean_recovery":"baseline";int wait=-1,wait_frames=-1,accepted_since_gap=0,issued=0,observed_issued=0;
      int wrong_matches=0,missing_observations=0;bool kept=true,present=true;
      for(int i=clip.warmup;i<finish;++i){const auto& state=result[size_t(i)];
        const TrackOutput* t=by_id(state,target_id);
        present=present&&bool(t);
        if(i>=clip.warmup+missing){
          missing_observations+=!t||!t->measurement_used;
          wrong_matches+=t&&t->measurement_used&&t->detection_index!=frames[clip.first+size_t(i)].reference_index;
        }
        if(!t||(i>=clip.warmup+missing&&(!t->measurement_used||
              t->detection_index!=frames[clip.first+size_t(i)].reference_index)))kept=false;
        if(i>=clip.warmup+missing&&t&&t->measurement_used){++accepted_since_gap;if(wait<0&&t->future_valid){
          wait=accepted_since_gap;wait_frames=i-(clip.warmup+missing)+1;}}
        const bool valid=t&&t->future_valid,observed_valid=valid&&t->measurement_used;
        issued+=valid;observed_issued+=observed_valid;++total.opportunities;
        const auto& reference_frame=frames.at(clip.first+size_t(i+clip.lead));
        const auto reference_center=reference_frame.observations.at(size_t(reference_frame.reference_index)).center;
        const auto& current_frame=frames[clip.first+size_t(i)];
        const bool moving=cv::norm(reference_center-current_frame.observations.at(size_t(current_frame.reference_index)).center)>2;
        if(i>=clip.warmup+missing&&moving){++total.moving_opportunities;total.moving_observed_issued+=observed_valid;}
        rows<<case_id<<','<<missing<<','<<name<<','<<frames[clip.first+size_t(i)].index<<','
          <<(i<clip.warmup+missing?"masked":"recovery")<<','<<bool(t)<<','<<(t?t->id:0)<<','
          <<(t&&t->measurement_used)<<','<<valid<<','<<observed_valid<<','<<(t?t->forecast.blocks:0)<<','
          <<(t&&t->clean_recovery_used)<<','<<reference_frame.index<<','<<reference_center.x<<','<<reference_center.y<<',';
        if(t)rows<<t->future_center.x<<','<<t->future_center.y<<',';else rows<<",,";
        if(valid){const double error=cv::norm(t->future_center-reference_center);rows<<error;
          if(observed_valid){total.errors.push_back(error);
            if(moving)total.moving_errors.push_back(error);
            const auto* b=by_id(baseline[size_t(i)],target_id);if(mode&&(!b||!b->future_valid||!b->measurement_used)){
              total.added.push_back(error);if(moving)total.added_moving.push_back(error);}}}
        rows<<'\n';
      }
      ++total.cases;total.id_kept+=kept;total.model_issued+=issued;total.observed_issued+=observed_issued;
      total.id_present+=present;total.mature_before_mask+=mature_before_mask;
      total.wrong_reference_matches+=wrong_matches;total.missing_recovery_observations+=missing_observations;
      if(wait>=0){++total.recovered;total.wait_sum+=wait;total.wait_frames_sum+=wait_frames;}
      case_results<<case_id<<','<<missing<<','<<name<<','<<target_id<<','<<mature_before_mask<<','<<present<<','<<kept<<','
        <<wrong_matches<<','<<missing_observations<<','<<wait<<','<<wait_frames<<','<<issued<<','<<observed_issued<<','<<finish-clip.warmup<<'\n';
    }
  }
  manifest.close();rows.close();case_results.close();require(bool(manifest)&&bool(rows)&&bool(case_results),"replay audit write failed");
  std::ostringstream report;report<<std::setprecision(8)<<"frames="<<frames.size()<<" fps="<<source.fps()<<" clips="<<selected.size()
    <<" scenarios="<<case_id<<" warmup_frames="<<warmup<<" recovery_frames="<<recovery
    <<" max_gap_seconds="<<candidate.clean_recovery_max_gap_seconds<<" recovery_hits="<<candidate.clean_recovery_hits
    <<" max_nis="<<candidate.clean_recovery_max_nis<<" measurement_quality_enabled=0\n";
  for(int missing:{1,2,3})for(int mode=0;mode<2;++mode){const auto& s=totals[missing-1][mode];
    report<<"missing="<<missing<<" mode="<<(mode?"clean_recovery":"baseline")<<" cases="<<s.cases
      <<" mature_before_mask="<<s.mature_before_mask<<" id_present="<<s.id_present
      <<" reference_association_complete="<<s.id_kept<<" wrong_reference_matches="<<s.wrong_reference_matches
      <<" missing_recovery_observations="<<s.missing_recovery_observations
      <<" recovered="<<s.recovered<<" mean_recovery_accepted_frames="<<(s.recovered?s.wait_sum/s.recovered:-1)
      <<" mean_recovery_frames="<<(s.recovered?s.wait_frames_sum/s.recovered:-1)
      <<" model_issued="<<s.model_issued<<'/'<<s.opportunities<<" observed_issued="<<s.observed_issued<<'/'<<s.cases*recovery
      <<" observed_reference_error_px "<<stats(s.errors)<<" additional_reference_error_px "<<stats(s.added)
      <<" moving_observed_issued="<<s.moving_observed_issued<<'/'<<s.moving_opportunities
      <<" moving_reference_error_px "<<stats(s.moving_errors)<<" additional_moving_reference_error_px "<<stats(s.added_moving)<<'\n';}
  report<<"Frozen, nonoverlapping clips are selected by continuous unmodified local ID; all 1/2/3-frame masks use the same clips.\n"
    <<"Each unmodified frame chooses its first accepted local ID; all detector candidates enter replay. Audit target ID is fixed before masking, never repaired afterward.\n"
    <<"Errors use withheld/future detector observations retrospectively, NOT manual or physical ground truth.\n"
    <<"Moving diagnostic means unmodified detector displacement over the forecast horizon exceeds 2 px; it is retrospective, not a tracking gate.\n"
    <<"Model coast forecasts are not drawn observations. Visibility is unchanged; observed_issued counts accepted-observation forecasts only.\n"
    <<"Recovery waiting counts accepted hits; unrecovered cases are -1 and retained in case_results.csv.\n"
    <<"No forecast state, ID, association, covariance or geometry differs between policies on any replay step.\n"
    <<"state=completed\n";
  std::ofstream summary(output/"report.txt");summary<<report.str();summary.close();require(bool(summary),"cannot write replay report");
  std::cout<<report.str();
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
