#include "core.hpp"
#include "detection_backend.hpp"
#include "estimator.hpp"
#include "prediction_audit.hpp"
#include "shape_forecast.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
using namespace outpost;
struct Shapes{Quad baseline,proposed;bool candidate=false;cv::Point2f predicted_center;bool selected=false;};
struct Errors{std::vector<double> base,proposed,base_shape,proposed_shape,base_iou,proposed_iou;};
double corner_error(const Quad& a,const Quad& b,cv::Point2f offset={}){
  double sum=0;for(int k=0;k<4;++k)sum+=cv::norm(a[k]-b[k]-offset);return sum/4;
}
double iou(const Quad& a,const Quad& b){
  std::vector<cv::Point2f> x(a.begin(),a.end()),y(b.begin(),b.end()),intersection;
  const double area=cv::intersectConvexConvex(x,y,intersection,true);
  return area/(cv::contourArea(x)+cv::contourArea(y)-area);
}
void stats(const std::string& name,std::vector<double> values){
  if(values.empty()){std::cout<<name<<" n=0\n";return;}
  std::sort(values.begin(),values.end());double sum=0;for(double v:values)sum+=v;
  std::cout<<std::fixed<<std::setprecision(4)<<name<<" n="<<values.size()<<" mean="<<sum/values.size()
           <<" p95="<<values[size_t(std::ceil(.95*values.size()))-1]<<" max="<<values.back()<<'\n';
}
int main(int argc,char**argv){try{
  if(argc!=4){std::cerr<<"usage: shape_evaluation video.yaml output.csv shape_settings.yaml\n";return 2;}
  const auto c=Config::load(argv[1]);
  require_estimator_mode(c,"image_2d","shape_evaluation");
  ShapeForecastConfig config;
  const auto settings=YAML::LoadFile(argv[3]);
  config.history=settings["history_samples"].as<int>();
  config.trend_sigma=settings["trend_sigma"].as<double>();
  config.max_scale=settings["max_scale"].as<double>();
  config.evidence_samples=settings["evidence_min_samples"].as<int>();
  config.evidence_window=settings["evidence_window"].as<int>();
  config.evidence_min_gain=settings["evidence_min_gain_px"].as<double>();
  const auto mode=settings["mode"].as<std::string>(),selection=settings["selection"].as<std::string>();
  if(mode!="width"&&mode!="full")throw std::runtime_error("invalid shape experiment mode");
  if(selection!="direct"&&selection!="past_evidence")throw std::runtime_error("invalid selection policy");
  config.width_only=mode=="width";const bool gated=selection=="past_evidence";
  if(!shape_config_valid(config))throw std::runtime_error("invalid shape experiment parameters");
  VideoSource source(c);auto detector=make_detector(c.detector.backend,c.detector);
  std::unique_ptr<Estimator> target;
  source.forecast_frame(0,c.prediction_seconds);
  if(std::filesystem::exists(argv[2]))throw std::runtime_error("output exists");
  std::ofstream log(argv[2]);if(!log)throw std::runtime_error("cannot create shape audit");
  log<<"frame,target_frame,id,status,candidate_available,baseline_corner_px,proposed_corner_px,baseline_shape_px,proposed_shape_px,baseline_iou,proposed_iou,selected\n";
  PredictionAudit audit;std::map<std::uint64_t,std::vector<TimedShape>> histories;
  std::map<std::uint64_t,ShapeEvidence> evidence;
  std::map<std::pair<long,std::uint64_t>,Shapes> pending;
  std::map<std::string,int> counts;std::map<std::string,Errors> groups;
  Frame frame;int issued=0,offered=0,selected=0;
  auto finish=[&](const AuditedForecast& p,const std::string& status,const TrackOutput* observed){
    ++counts[status];const auto key=std::make_pair(p.frame,p.id);const auto shape=pending.at(key);pending.erase(key);
    log<<p.frame<<','<<p.target<<','<<p.id<<','<<status<<','<<shape.candidate;
    if(observed){
      const auto& chosen=shape.selected?shape.proposed:shape.baseline;
      const double be=corner_error(shape.baseline,observed->current),pe=corner_error(chosen,observed->current);
      const auto offset=shape.predicted_center-observed->current_center;
      const double bs=corner_error(shape.baseline,observed->current,offset),ps=corner_error(chosen,observed->current,offset);
      if(shape.candidate)evidence[p.id].add(bs-corner_error(shape.proposed,observed->current,offset),config);
      const double bi=iou(shape.baseline,observed->current),pi=iou(chosen,observed->current);
      log<<','<<be<<','<<pe<<','<<bs<<','<<ps<<','<<bi<<','<<pi;
      for(const auto& name:{std::string("all"),std::string(shape.candidate?"candidate":"fallback"),
                           std::string(p.frame<450?"before_450":"from_450")}){
        auto& g=groups[name];g.base.push_back(be);g.proposed.push_back(pe);g.base_shape.push_back(bs);
        g.proposed_shape.push_back(ps);g.base_iou.push_back(bi);g.proposed_iou.push_back(pi);
      }
    }else log<<",,,,,,";
    log<<','<<shape.selected<<'\n';
  };
  std::optional<ImagePreprocessor> preprocessing;
  while(source.next(frame)){
    if(!preprocessing)preprocessing.emplace(c.geometry,frame.image.size());
    frame.image=preprocessing->process(frame.image);
    const auto detected=detector->detect(frame.image);
    if(!target)target=make_estimator(c,frame.image.size());
    const auto estimate=target->update(frame.time_seconds,detected.observations,c.prediction_seconds).image;
    const auto& tracks=estimate.tracks;
    const auto decisions=visibility_batch(tracks,detected.lamps,detected.pairs,c.visibility);
    std::map<std::uint64_t,const TrackOutput*> observed;for(const auto& t:tracks)if(t.measurement_used)observed[t.id]=&t;
    for(auto it=evidence.begin();it!=evidence.end();)if(!observed.count(it->first))it=evidence.erase(it);else ++it;
    audit.observe(frame.index,observed,finish);
    for(auto it=histories.begin();it!=histories.end();)if(!observed.count(it->first))it=histories.erase(it);else ++it;
    for(size_t i=0;i<tracks.size();++i){const auto& t=tracks[i];if(!t.measurement_used)continue;
      auto& h=histories[t.id];h.push_back({frame.time_seconds,t.current_center,t.current});
      if(h.size()>size_t(config.history))h.erase(h.begin());
      if(!decisions[i].future)continue;
      const auto proposed=forecast_shape(h,t.future_center,c.prediction_seconds,config);
      const bool use=bool(proposed)&&(!gated||evidence[t.id].prefer_candidate(config));
      pending[{frame.index,t.id}]={t.future,proposed.value_or(t.future),bool(proposed),t.future_center,use};
      const long future=source.forecast_frame(frame.index,c.prediction_seconds);
      const AuditedForecast forecast{frame.index,future,t.id,t.current_center,t.future_center,{},true};
      if(future<0)finish(forecast,future==-2?"target_not_reached":"timestamp_unmatched",nullptr);
      else audit.issue(forecast);
      ++issued;if(proposed)++offered;if(use)++selected;
    }
  }
  audit.finish(finish);log.flush();if(!log||!pending.empty())throw std::runtime_error("incomplete shape audit");
  std::cout<<"issued="<<issued<<" candidate_available="<<offered<<" history="<<config.history
           <<" mode="<<(config.width_only?"width":"full")<<" gated="<<gated<<" selected="<<selected<<'\n';
  for(const auto& [status,n]:counts)std::cout<<status<<'='<<n<<'\n';
  for(const auto& [name,g]:groups){stats(name+" baseline_corner",g.base);stats(name+" proposed_corner",g.proposed);
    stats(name+" baseline_shape_only",g.base_shape);stats(name+" proposed_shape_only",g.proposed_shape);
    stats(name+" baseline_iou",g.base_iou);stats(name+" proposed_iou",g.proposed_iou);}
  std::cout<<"Detector-relative scores, not manual ground truth. Rejections use the existing observed-shape translation for comparison only.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
