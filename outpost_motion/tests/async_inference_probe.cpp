#include "core.hpp"
#include "yolov8_backend.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace outpost;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void rejected(F action){bool failed=false;try{action();}catch(const std::exception&){failed=true;}require(failed,"invalid async operation accepted");}
int main(int argc,char** argv){try{
  if(argc!=3&&argc!=4)throw std::runtime_error("usage: async_inference_probe config.yaml NEW_OUTPUT_DIR [frame_limit]");
  const int limit=argc==4?std::stoi(argv[3]):0;if(limit<0)throw std::runtime_error("negative frame limit");
  const auto config=Config::load(argv[1]);
  require(std::filesystem::create_directory(argv[2]),"output directory already exists or cannot be created");
  std::ofstream log(std::filesystem::path(argv[2])/"async.csv"),summary(std::filesystem::path(argv[2])/"summary.csv");
  require(bool(log)&&bool(summary),"cannot create probe logs");
  log<<std::setprecision(17)<<"capacity,frame,time,latency_ms,max_corner_difference_px\n";
  summary<<std::setprecision(17)<<"capacity,frames,total_ms,mean_delivery_latency_ms,p95_delivery_latency_ms,max_corner_difference_px\n";
  YoloV8Backend backend(config.detector.neural_model,config.detector.neural_score,config.detector.neural_nms,config.detector.neural_inference_threads);
  rejected([&]{backend.configure_async(0);});rejected([&]{backend.configure_async(5);});
  backend.configure_async(1);rejected([&]{backend.collect();});
  cv::Mat fixture(64,64,CV_8UC3,cv::Scalar());backend.submit(0,0,fixture);
  rejected([&]{backend.submit(1,.03,fixture);});rejected([&]{backend.configure_async(2);});
  rejected([&]{backend.infer_candidates(fixture);});backend.collect();
  rejected([&]{backend.submit(0,.04,fixture);});rejected([&]{backend.submit(1,0,fixture);});
  rejected([&]{backend.submit(1,.04,cv::Mat());});
  std::vector<std::vector<NeuralCandidate>> baseline;std::vector<double> times,serial_latency;
  auto begin=DetectionClock::now();
  {VideoSource source(config);Frame frame;std::optional<ImagePreprocessor> preprocessing;
    while((!limit||int(baseline.size())<limit)&&source.next(frame)){
      if(!preprocessing)preprocessing.emplace(config.geometry,frame.image.size());
      frame.image=preprocessing->process(frame.image);
      const auto start=DetectionClock::now();backend.infer_candidates(frame.image);
      serial_latency.push_back(detection_ms(start,DetectionClock::now()));
      baseline.push_back(backend.candidates());times.push_back(frame.time_seconds);
    }
  }
  require(!baseline.empty(),"empty probe input");
  auto record=[&](int capacity,double total,std::vector<double> latency,double max_error){
    require(latency.size()==baseline.size(),"lost async frames");std::sort(latency.begin(),latency.end());
    double sum=0;for(double v:latency)sum+=v;
    summary<<capacity<<','<<latency.size()<<','<<total<<','<<sum/latency.size()<<','
      <<latency[size_t(std::ceil(.95*latency.size()))-1]<<','<<max_error<<'\n';
    std::cout<<"capacity="<<capacity<<" frames="<<latency.size()<<" total_ms="<<total
      <<" mean_delivery_latency_ms="<<sum/latency.size()<<" max_corner_difference_px="<<max_error<<'\n';
  };
  record(0,detection_ms(begin,DetectionClock::now()),serial_latency,0);
  for(int capacity:{1,2}){
    backend.configure_async(capacity);std::vector<double> latency;size_t consumed=0;double max_error=0;
    auto consume=[&]{const auto result=backend.collect();
      require(result.frame==long(consumed)&&result.time==times.at(consumed),"async frame/time reordered");
      const auto& expected=baseline.at(consumed);require(expected.size()==result.candidates.size(),"async candidate count changed");
      double frame_error=0;
      for(size_t i=0;i<expected.size();++i){const auto& a=expected[i];const auto& b=result.candidates[i];
        require(a.raw_class==b.raw_class&&std::abs(a.confidence-b.confidence)<1e-6,"async class/confidence changed");
        for(int k=0;k<4;++k)frame_error=std::max(frame_error,cv::norm(a.detection.measurement_points[k]-b.detection.measurement_points[k]));
        require(cv::norm(a.box.tl()-b.box.tl())<.001&&cv::norm(a.box.br()-b.box.br())<.001,"async box changed");
      }
      max_error=std::max(max_error,frame_error);require(frame_error<.001,"async corners changed");
      latency.push_back(result.latency_ms);log<<capacity<<','<<result.frame<<','<<result.time<<','<<result.latency_ms<<','<<frame_error<<'\n';++consumed;
    };
    begin=DetectionClock::now();
    {VideoSource source(config);Frame frame;size_t submitted=0;std::optional<ImagePreprocessor> preprocessing;
      while((!limit||int(submitted)<limit)&&source.next(frame)){
        if(backend.pending()==size_t(capacity))consume();
        if(!preprocessing)preprocessing.emplace(config.geometry,frame.image.size());
        frame.image=preprocessing->process(frame.image);backend.submit(frame.index,frame.time_seconds,frame.image);++submitted;
        // Verify requests do not depend on the caller's mutable image storage.
        frame.image.setTo(cv::Scalar());
      }
      while(backend.pending())consume();
    }
    record(capacity,detection_ms(begin,DetectionClock::now()),latency,max_error);
  }
  log.flush();summary.flush();require(bool(log)&&bool(summary),"incomplete probe output");
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
