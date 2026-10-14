#include "core.hpp"
#include "detection_correspondence.hpp"
#include "yolov8_backend.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
using namespace outpost;
int main(int argc,char**argv){try{
  if(argc!=2){std::cerr<<"usage: neural_probe experiment.yaml\n";return 2;}
  const auto experiment=YAML::LoadFile(argv[1]);
  const auto c=Config::load(experiment["video_config"].as<std::string>());
  const double fraction=experiment["endpoint_fraction"].as<double>();
  detection_correspondence({}, {}, fraction);
  const auto brightness_levels=experiment["diagnostic_brightness_levels"].as<std::vector<int>>();
  if(brightness_levels.empty())throw std::runtime_error("diagnostic brightness levels empty");
  std::vector<TraditionalBackend> diagnostic_detectors;
  for(int level:brightness_levels){
    if(level<0||level>255)throw std::runtime_error("diagnostic brightness outside [0,255]");
    auto config=c.detector;config.brightness=level;diagnostic_detectors.emplace_back(config);
  }
  const std::filesystem::path out(experiment["output_directory"].as<std::string>());
  if(std::filesystem::exists(out))throw std::runtime_error("output directory exists");
  YoloV8Backend neural(experiment["model_xml"].as<std::string>(),experiment["score_threshold"].as<double>(),experiment["nms_threshold"].as<double>());
  std::filesystem::create_directories(out);std::ofstream log(out/"neural_candidates.csv"),counts(out/"frames.csv");
  std::filesystem::create_directory(out/"disagreements");
  std::ofstream matches(out/"correspondence.csv");
  std::ofstream sweep(out/"brightness_diagnostic.csv");
  if(!log||!counts||!matches||!sweep)throw std::runtime_error("cannot create logs");
  sweep<<"frame,brightness,traditional_candidates,matched,baseline_unmatched_supported,baseline_matched_lost\n";
  matches<<"frame,neural_index,status,traditional_index,max_endpoint_error_px\n";
  log<<"frame,index,raw_class,confidence,cx,cy,tlx,tly,trx,try,brx,bry,blx,bly\n";
  counts<<"frame,traditional,neural,inference_ms,matched,neural_unmatched,ambiguous,traditional_unmatched\n";
  TraditionalBackend traditional(c.detector);VideoSource source(c);Frame f;
  std::optional<ImagePreprocessor> pre;long frames=0,total=0,old=0,matched_total=0,ambiguous_total=0;double milliseconds=0;
  while(source.next(f)){if(!pre)pre.emplace(c.geometry,f.image.size());f.image=pre->process(f.image);
    auto a=traditional.detect(f.image);auto start=std::chrono::steady_clock::now();auto b=neural.detect(f.image);
    const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    milliseconds+=elapsed;++frames;total+=long(b.observations.size());old+=long(a.observations.size());
    const auto correspondence=detection_correspondence(b.observations,a.observations,fraction);
    for(size_t k=0;k<diagnostic_detectors.size();++k){
      const auto diagnostic=diagnostic_detectors[k].detect(f.image);
      const auto compared=detection_correspondence(b.observations,diagnostic.observations,fraction);
      size_t supported=0,recovered=0,lost=0;
      for(size_t i=0;i<compared.size();++i){
        const bool before=correspondence[i].traditional>=0,after=compared[i].traditional>=0;
        supported+=after;recovered+=!before&&after;lost+=before&&!after;
      }
      sweep<<f.index<<','<<brightness_levels[k]<<','<<diagnostic.observations.size()<<','<<supported<<','<<recovered<<','<<lost<<'\n';
    }
    size_t matched=0,ambiguous=0;
    for(size_t i=0;i<correspondence.size();++i){const auto& r=correspondence[i];
      matches<<f.index<<','<<i<<',';
      if(r.traditional>=0){++matched;matches<<"matched,"<<r.traditional<<','<<r.max_endpoint_error;}
      else{ambiguous+=r.ambiguous;matches<<(r.ambiguous?"ambiguous,,":"unmatched,,");
        // Evidence only: raw pixels are shown separately; drawing cannot conceal
        // absent or merged lamps. Fixed-size crop does not affect inference.
        const auto& d=b.observations[i];
        const int x=std::clamp(cvRound(d.center.x)-80,0,std::max(0,f.image.cols-160));
        const int y=std::clamp(cvRound(d.center.y)-60,0,std::max(0,f.image.rows-120));
        const cv::Rect roi(x,y,std::min(160,f.image.cols),std::min(120,f.image.rows));
        cv::Mat raw=f.image(roi).clone(),network=raw.clone(),geometry=raw.clone();
        auto draw=[&](cv::Mat& dst,const auto& points,cv::Scalar color){
          std::vector<cv::Point> p;for(auto q:points)p.emplace_back(cvRound(q.x)-x,cvRound(q.y)-y);
          cv::polylines(dst,p,true,color,1,cv::LINE_AA);
        };
        draw(network,d.measurement_points,{255,0,255});
        for(const auto& lamp:a.lamps)draw(geometry,lamp.vertices,{0,255,255});
        for(const auto& observed:a.observations)draw(geometry,observed.measurement_points,{0,255,0});
        cv::Mat sheet;std::vector<cv::Mat> panels{raw,network,geometry};cv::hconcat(panels,sheet);
        cv::resize(sheet,sheet,{},2,2,cv::INTER_NEAREST);
        cv::putText(sheet,"raw | neural | lamps(yellow), accepted(green)",{4,16},cv::FONT_HERSHEY_SIMPLEX,.4,{255,255,255});
        const auto filename="frame_"+std::to_string(f.index)+"_candidate_"+std::to_string(i)+".png";
        if(!cv::imwrite((out/"disagreements"/filename).string(),sheet))throw std::runtime_error("disagreement image write failed");
      }
      matches<<'\n';
    }
    matched_total+=matched;ambiguous_total+=ambiguous;
    counts<<f.index<<','<<a.observations.size()<<','<<b.observations.size()<<','<<elapsed<<','<<matched<<','
          <<b.observations.size()-matched-ambiguous<<','<<ambiguous<<','<<a.observations.size()-matched<<'\n';
    for(size_t i=0;i<neural.candidates().size();++i){const auto& d=neural.candidates()[i];
      log<<f.index<<','<<i<<','<<d.raw_class<<','<<d.confidence<<','<<d.detection.center.x<<','<<d.detection.center.y;
      for(auto p:d.detection.measurement_points)log<<','<<p.x<<','<<p.y;log<<'\n';}
    if(f.index==180||f.index==360||f.index==480||f.index==720){
      for(const auto& d:b.observations){std::vector<cv::Point> p;for(auto q:d.corners)p.emplace_back(cvRound(q.x),cvRound(q.y));
        cv::polylines(f.image,p,true,{255,0,255},1,cv::LINE_AA);}
      cv::putText(f.image,"Experimental neural keypoints; NOT lamp-edge enclosure",{5,20},cv::FONT_HERSHEY_SIMPLEX,.4,{255,255,255});
      if(!cv::imwrite((out/("frame_"+std::to_string(f.index)+".png")).string(),f.image))throw std::runtime_error("image write failed");
    }
  }
  log.flush();counts.flush();matches.flush();sweep.flush();if(!log||!counts||!matches||!sweep||!frames)throw std::runtime_error("incomplete probe");
  std::cout<<"frames="<<frames<<" traditional_candidates="<<old<<" neural_candidates="<<total
           <<" mean_inference_ms="<<milliseconds/frames<<" matched="<<matched_total<<" ambiguous="<<ambiguous_total
           <<"\nCounts are not precision/recall; no digit classifier or physical ID; not promoted to default.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
