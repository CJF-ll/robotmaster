#include "core.hpp"
#include "detection_backend.hpp"
#include "estimator.hpp"
#include "prediction_fit.hpp"
#include "prediction_audit.hpp"
#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <sstream>
using namespace outpost;
struct Errors {
  std::vector<double> live,still,challenger,paired_live;
  void print(const std::string& label,const std::string& model)const{
    auto stats=[](std::vector<double> v){
      if(v.empty())return std::string("n=0");
      std::sort(v.begin(),v.end());double sum=0,squares=0;for(double e:v){sum+=e;squares+=e*e;}
      std::ostringstream s;s<<std::fixed<<std::setprecision(3)<<"n="<<v.size()<<" mean="<<sum/v.size()
        <<" median="<<v[v.size()/2]<<" p95="<<v[size_t(std::ceil(.95*v.size()))-1]
        <<" rmse="<<std::sqrt(squares/v.size())<<" max="<<v.back();return s.str();
    };
    std::cout<<label<<" live "<<stats(live)<<"\n"<<label<<" hold_position "<<stats(still)<<"\n"
             <<label<<' '<<model<<"_shadow "<<stats(challenger)<<"\n"
             <<label<<" live_on_same_shadow_samples "<<stats(paired_live)<<"\n";
  }
};
struct ConfidenceSummary {
  int n=0,covered=0;double error_sum=0,sigma_sum=0;
  void add(double error,double nis,double sigma){++n;covered+=nis<=5.991464547107979;
    error_sum+=error;sigma_sum+=sigma;}
};
int main(int argc,char**argv){try{
  if(argc<3||argc>7){std::cerr<<"usage: prediction_evaluation config.yaml output.csv [history_samples] [quadratic|linear|stationary] [stationary_duration_seconds stationary_diameter_px]\n";return 2;}
  const int window=argc>=4?std::stoi(argv[3]):9;
  const std::string model=argc>=5?argv[4]:"quadratic";
  if(model!="quadratic"&&model!="linear"&&model!="stationary")throw std::runtime_error("unknown shadow model");
  if((model=="stationary"&&argc!=7)||(model!="stationary"&&argc>5))
    throw std::runtime_error("stationary model requires explicit duration and diameter");
  const double duration=model=="stationary"?std::stod(argv[5]):0;
  const double diameter=model=="stationary"?std::stod(argv[6]):0;
  if(model=="stationary"&&(!std::isfinite(duration)||duration<=0||!std::isfinite(diameter)||diameter<0))
    throw std::runtime_error("invalid stationary thresholds");
  if(window<5||window>60)throw std::runtime_error("history_samples must be 5..60");
  auto c=Config::load(argv[1]);
  require_estimator_mode(c,"image_2d","prediction_evaluation");
  VideoSource source(c);
  source.forecast_frame(0,c.prediction_seconds); // Validate time policy before opening outputs.
  // Do not overwrite previous diagnostics without an explicit different path.
  if(std::filesystem::exists(argv[2]))throw std::runtime_error("output already exists");
  const std::string confidence_path=std::string(argv[2])+".confidence.csv";
  if(std::filesystem::exists(confidence_path))throw std::runtime_error("confidence output already exists");
  std::ofstream confidence(confidence_path);
  if(!confidence)throw std::runtime_error("cannot create confidence audit");
  confidence<<std::setprecision(17)<<"frame,target_frame,id,status,consecutive_hits,filtered_x,filtered_y,vx,vy,velocity_var_x,velocity_var_y,future_var_x,future_var_y,error_px,residual_nis,inside_nominal_95,target_measurement_variance\n";
  std::ofstream log(argv[2]);if(!log)throw std::runtime_error("cannot create evaluation CSV");
  log<<std::setprecision(10)<<"frame,target_frame,id,status,origin_x,origin_y,predicted_x,predicted_y,observed_x,observed_y,error_px,hold_error_px,shadow_error_px\n";
  auto detector=make_detector(c.detector.backend,c.detector);
  std::unique_ptr<Estimator> target;
  std::map<std::uint64_t,std::vector<TimedCenter>> history;
  PredictionAudit audit;std::map<std::string,int> status_counts;
  std::map<std::string,Errors> groups;Frame frame;long count=0,issued=0;
  std::map<std::pair<long,std::uint64_t>,TrackOutput> issued_states;
  std::map<std::string,ConfidenceSummary> confidence_groups;
  auto finish=[&](const AuditedForecast& p,const std::string& status,const TrackOutput* observed){
    const auto key=std::make_pair(p.frame,p.id);
    const auto state=issued_states.at(key);issued_states.erase(key);
    confidence<<p.frame<<','<<p.target<<','<<p.id<<','<<status<<','<<state.consecutive_hits<<','
      <<state.filtered_center.x<<','<<state.filtered_center.y<<','<<state.velocity.x<<','<<state.velocity.y<<','
      <<state.velocity_variance[0]<<','<<state.velocity_variance[1]<<','
      <<state.future_position_variance[0]<<','<<state.future_position_variance[1];
    ++status_counts[status];log<<p.frame<<','<<p.target<<','<<p.id<<','<<status<<','
      <<p.origin.x<<','<<p.origin.y<<','<<p.predicted.x<<','<<p.predicted.y;
    if(status=="scored"){
      const auto truth=observed->current_center; // Detector observation, NOT hand-labeled truth.
      const double e=cv::norm(p.predicted-truth),base=cv::norm(p.origin-truth);
      const double challenger=p.challenger?cv::norm(*p.challenger-truth):0;
      // Future detector measurement includes its own R, in addition to predicted P.
      const auto residual=p.predicted-truth;
      const double sx=state.future_position_variance[0]+observed->measurement_variances[0];
      const double sy=state.future_position_variance[1]+observed->measurement_variances[1];
      if(!std::isfinite(sx)||!std::isfinite(sy)||sx<=0||sy<=0)throw std::runtime_error("invalid forecast variance");
      const double normalized=residual.x*residual.x/sx+residual.y*residual.y/sy;
      confidence<<','<<e<<','<<normalized<<','<<(normalized<=5.991464547107979)<<','<<observed->measurement_variance;
      const std::string age=state.consecutive_hits<5?"hits_below_5":state.consecutive_hits==5?"hits_5":
                             state.consecutive_hits<10?"hits_6_to_9":"hits_10_plus";
      for(const auto& group:{std::string("all"),age,std::string(base>2?"moving":"low_displacement"),
                            age+std::string(base>2?"/moving":"/low_displacement")})
        confidence_groups[group].add(e,normalized,std::sqrt(std::max(sx,sy)));
      log<<','<<truth.x<<','<<truth.y<<','<<e<<','<<base<<',';
      if(p.challenger)log<<challenger;
      for(const auto& group:{std::string("all"),std::string(base>2?"displacement_gt_2px":"displacement_le_2px"),
                            std::string(p.frame<450?"frames_before_450":"frames_from_450")}){
        auto& g=groups[group];g.live.push_back(e);g.still.push_back(base);
        if(p.challenger){g.challenger.push_back(challenger);g.paired_live.push_back(e);}
      }
    }else {log<<",,,,,";confidence<<",,,,";}
    confidence<<'\n';
    log<<'\n';
  };
  std::optional<ImagePreprocessor> preprocessing;
  while(source.next(frame)){
    if(!preprocessing)preprocessing.emplace(c.geometry,frame.image.size());
    frame.image=preprocessing->process(frame.image);
    ++count;const auto detected=detector->detect(frame.image);
    if(!target)target=make_estimator(c,frame.image.size());
    const auto estimate=target->update(frame.time_seconds,detected.observations,c.prediction_seconds);
    const auto& tracks=estimate.image.tracks;
    std::map<std::uint64_t,const TrackOutput*> observed;
    for(const auto& t:tracks)if(t.measurement_used)observed[t.id]=&t;
    for(auto it=history.begin();it!=history.end();)if(!observed.count(it->first))it=history.erase(it);else ++it;
    audit.observe(frame.index,observed,finish);
    for(size_t i=0;i<tracks.size();++i){const auto& t=tracks[i];if(!t.measurement_used)continue;
      auto& h=history[t.id];h.push_back({frame.time_seconds,t.current_center});
      if(int(h.size())>window)h.erase(h.begin());
      // Score center state independently of silhouette/render permissions.
      if(!t.future_center_valid)continue;
      std::optional<cv::Point2f> shadow;
      if(model=="stationary")shadow=stationary_history(h,duration,diameter)?t.current_center:t.future_center;
      else if(int(h.size())==window)shadow=model=="linear"?linear_forecast(h,c.prediction_seconds):
                                          quadratic_forecast(h,c.prediction_seconds);
      issued_states.emplace(std::make_pair(frame.index,t.id),t);
      const long future=source.forecast_frame(frame.index,c.prediction_seconds);
      const AuditedForecast forecast{frame.index,future,t.id,t.current_center,t.future_center,shadow,true};++issued;
      if(future<0)finish(forecast,future==-2?"target_not_reached":"timestamp_unmatched",nullptr);
      else audit.issue(forecast);
    }
  }
  audit.finish(finish);
  confidence.flush();
  if(!confidence||!issued_states.empty())throw std::runtime_error("confidence audit incomplete");
  log.flush();if(!log)throw std::runtime_error("failed to write complete audit CSV");
  int resolved=0;for(const auto& [status,n]:status_counts){resolved+=n;std::cout<<status<<'='<<n<<' ';}
  if(resolved!=issued)throw std::runtime_error("forecast accounting mismatch");
  std::cout<<"\nframes="<<count<<" issued="<<issued<<" horizon="<<c.prediction_seconds<<" history="<<window<<'\n';
  for(const auto& [name,errors]:groups)errors.print(name,model);
  for(const auto& [name,g]:confidence_groups)std::cout<<"confidence "<<name<<" n="<<g.n
    <<" mean_error="<<g.error_sum/g.n<<" mean_residual_sigma="<<g.sigma_sum/g.n
    <<" nominal95_coverage="<<double(g.covered)/g.n<<'\n';
  std::cout<<"Confidence is model-conditional; 95% is nominal 2D Gaussian coverage, not a calibrated guarantee.\n";
  std::cout<<"Errors compare detector observations on uninterrupted local IDs, not physical ground truth.\n"
              "Unscored forecasts remain in CSV; fitted model is shadow-only and does not affect live tracking.\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
