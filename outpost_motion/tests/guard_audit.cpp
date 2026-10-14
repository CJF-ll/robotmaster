#include "core.hpp"
#include "detection_backend.hpp"
#include "target_model.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
using namespace outpost;
struct Snapshot{long frame;cv::Mat image;Detection detection;};
cv::Mat evidence(const cv::Mat& image,const Detection& d){
  const int x=std::clamp(cvRound(d.center.x)-80,0,std::max(0,image.cols-160));
  const int y=std::clamp(cvRound(d.center.y)-50,0,std::max(0,image.rows-100));
  cv::Rect roi(x,y,std::min(160,image.cols),std::min(100,image.rows));
  auto raw=image(roi).clone(),overlay=raw.clone();std::vector<cv::Point> points;
  for(auto p:d.measurement_points)points.emplace_back(cvRound(p.x)-x,cvRound(p.y)-y);
  cv::polylines(overlay,points,true,{0,255,255},1,cv::LINE_AA);
  cv::Mat result;std::vector<cv::Mat> parts{raw,overlay};cv::hconcat(parts,result);return result;
}
int main(int argc,char**argv){try{
 if(argc!=3){std::cerr<<"usage: guard_audit enabled_config.yaml new_output_directory\n";return 2;}
 auto config=Config::load(argv[1]);if(!config.tracker.one_sided_guard_enabled)throw std::runtime_error("guard must be enabled");
 std::filesystem::path out(argv[2]);if(std::filesystem::exists(out))throw std::runtime_error("output exists");
 VideoSource source(config);auto detector=make_detector(config.detector.backend,config.detector);
 auto model=make_target_model("image_2d",config.tracker,config.slots);
 std::filesystem::create_directories(out);std::ofstream log(out/"events.csv");if(!log)throw std::runtime_error("cannot create log");
 log<<"frame,id,candidate,previous_frame,old_left,old_right,new_left,new_right,left_ratio,right_ratio\n";
 std::map<std::uint64_t,Snapshot> history;Frame frame;std::optional<ImagePreprocessor> pre;int frames=0,events=0;
 while(source.next(frame)){++frames;if(!pre)pre.emplace(config.geometry,frame.image.size());frame.image=pre->process(frame.image);
   const auto detected=detector->detect(frame.image);const auto estimate=model->update(frame.time_seconds,detected.observations,config.prediction_seconds);
   for(const auto& diagnostic:estimate.associations){if(std::string(diagnostic.reason)!="one_sided_length_drop")continue;
     auto previous=history.find(diagnostic.id);if(previous==history.end()||diagnostic.detection<0||size_t(diagnostic.detection)>=detected.observations.size())throw std::runtime_error("missing guard provenance");
     const auto& old=previous->second;const auto& now=detected.observations[size_t(diagnostic.detection)];
     const auto& a=old.detection.measurement_points;const auto& b=now.measurement_points;
     const double al=cv::norm(a[3]-a[0]),ar=cv::norm(a[2]-a[1]),bl=cv::norm(b[3]-b[0]),br=cv::norm(b[2]-b[1]);
     log<<frame.index<<','<<diagnostic.id<<','<<diagnostic.detection<<','<<old.frame<<','<<al<<','<<ar<<','<<bl<<','<<br<<','<<bl/al<<','<<br/ar<<'\n';
     cv::Mat sheet;std::vector<cv::Mat> panels{evidence(old.image,old.detection),evidence(frame.image,now)};cv::hconcat(panels,sheet);cv::resize(sheet,sheet,{},2,2,cv::INTER_NEAREST);
     const auto title="previous "+std::to_string(old.frame)+" raw/points | rejected "+std::to_string(frame.index)+" raw/points";
     cv::putText(sheet,title,{3,15},cv::FONT_HERSHEY_SIMPLEX,.4,{255,255,255});
     if(!cv::imwrite((out/("frame_"+std::to_string(frame.index)+"_id_"+std::to_string(diagnostic.id)+".png")).string(),sheet))throw std::runtime_error("image write failed");
     ++events;
   }
   for(const auto& track:estimate.tracks)if(track.measurement_used){
     const auto& d=detected.observations.at(size_t(track.detection_index));history[track.id]={frame.index,frame.image.clone(),d};
   }
   // Bound retained evidence by currently output tracks, not video duration.
   for(auto it=history.begin();it!=history.end();){bool alive=false;for(auto t:estimate.tracks)alive|=t.id==it->first;
     if(!alive)it=history.erase(it);else ++it;}
 }
 log.flush();if(!log||!frames)throw std::runtime_error("incomplete replay");
 std::cout<<"frames="<<frames<<" rejected_events="<<events<<"\nEvidence only; length loss is not an occlusion label.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
