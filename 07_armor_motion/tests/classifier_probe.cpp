#include "core.hpp"
#include "classifier_contract.hpp"
#include <openvino/openvino.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <set>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
using namespace outpost;
struct Candidate {int index;std::array<cv::Point2f,4> points;};
int main(int argc,char**argv){try{
 if(argc!=2){std::cerr<<"usage: classifier_probe experiment.yaml\n";return 2;}
 auto yaml=YAML::LoadFile(argv[1]);const auto config=Config::load(yaml["video_config"].as<std::string>());
 const auto labels=yaml["labels"].as<std::vector<std::string>>();
 const double scale=yaml["crop_axis_half_scale"].as<double>(),threshold=yaml["confidence_threshold"].as<double>();
 if(labels.size()!=9||!std::isfinite(scale)||scale<=0||!std::isfinite(threshold)||threshold<0||threshold>1)throw std::runtime_error("invalid classifier configuration");
 // Offline perturbation only; never changes detector geometry or tracking.
 const double dx=yaml["diagnostic_shift_x_px"]?yaml["diagnostic_shift_x_px"].as<double>():0;
 const double dy=yaml["diagnostic_shift_y_px"]?yaml["diagnostic_shift_y_px"].as<double>():0;
 if(!std::isfinite(dx)||!std::isfinite(dy)||std::abs(dx)>16||std::abs(dy)>16)
   throw std::runtime_error("diagnostic shift must be finite and within +/-16 pixels");
 const std::string output=yaml["output_csv"].as<std::string>();if(std::filesystem::exists(output))throw std::runtime_error("output exists");
 const std::string evidence=yaml["evidence_dir"]?yaml["evidence_dir"].as<std::string>():"";
 std::set<long> evidence_frames;
 if(yaml["evidence_frames"])for(long f:yaml["evidence_frames"].as<std::vector<long>>()){
   if(f<0)throw std::runtime_error("negative evidence frame");evidence_frames.insert(f);
 }
 if(!evidence_frames.empty()&&evidence.empty())throw std::runtime_error("evidence_frames requires evidence_dir");
 if(!evidence.empty()&&std::filesystem::exists(evidence))throw std::runtime_error("evidence directory exists");
 std::ifstream candidates(yaml["candidates_csv"].as<std::string>());std::string line;
 if(!std::getline(candidates,line)||line!="frame,index,raw_class,confidence,cx,cy,tlx,tly,trx,try,brx,bry,blx,bly")throw std::runtime_error("invalid candidate CSV");
 std::map<long,std::vector<Candidate>> rows;size_t expected=0;
 while(std::getline(candidates,line)){std::stringstream stream(line);std::vector<std::string> fields;std::string v;
   while(std::getline(stream,v,','))fields.push_back(v);if(fields.size()!=14)throw std::runtime_error("bad candidate row");
   const long frame=std::stol(fields[0]);Candidate c{};c.index=std::stoi(fields[1]);if(frame<0||c.index<0)throw std::runtime_error("negative frame/index");
   for(auto old:rows[frame])if(old.index==c.index)throw std::runtime_error("duplicate candidate");
   for(int i=0;i<4;++i)c.points[i]={std::stof(fields[6+2*i]),std::stof(fields[7+2*i])};rows[frame].push_back(c);++expected;
 }
 if(!expected)throw std::runtime_error("no candidates");
 ov::Core core;auto graph=core.read_model(yaml["model_path"].as<std::string>());
 if(graph->inputs().size()!=1||graph->outputs().size()!=1||
    !graph->input().get_partial_shape().compatible(ov::PartialShape{1,1,32,32}))
   throw std::runtime_error("incompatible classifier input shape");
 // The supplied ONNX declares a dynamic batch. Specialize this offline replay
 // to batch one before checking concrete input/output contracts.
 graph->reshape(ov::Shape{1,1,32,32});
 if(graph->inputs().size()!=1||graph->outputs().size()!=1||graph->input().get_shape()!=ov::Shape{1,1,32,32}||graph->output().get_shape()!=ov::Shape{1,9}||graph->input().get_element_type()!=ov::element::f32||graph->output().get_element_type()!=ov::element::f32)throw std::runtime_error("expected f32 [1,1,32,32] -> [1,9]");
 auto model=core.compile_model(graph,"CPU",ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));auto request=model.create_infer_request();
 std::ofstream log(output);if(!log)throw std::runtime_error("cannot create output");
 log<<"frame,index,crop_clipped,label_id,label,confidence,above_threshold,status\n";
 std::ofstream details;
 if(!evidence.empty()){
   std::filesystem::create_directories(evidence);details.open(evidence+"/probabilities.csv");
   if(!details)throw std::runtime_error("cannot create evidence log");
   details<<"frame,index,crop_width,crop_height,status";
   for(int i=0;i<9;++i)details<<",p"<<i;
   details<<'\n'<<std::setprecision(10);
 }
 VideoSource source(config);Frame frame;std::optional<ImagePreprocessor> pre;
 size_t processed=0,invalid=0;std::map<std::string,int> totals,high;
 while(source.next(frame)){
   if(!pre)pre.emplace(config.geometry,frame.image.size());frame.image=pre->process(frame.image);
   const auto found=rows.find(frame.index);if(found==rows.end())continue;
   for(const auto& c:found->second){++processed;auto points=c.points;
     for(auto& point:points)point+=cv::Point2f(float(dx),float(dy));
     auto pattern=classifier_pattern(frame.image,points,scale);
     log<<frame.index<<','<<c.index<<','<<pattern.clipped<<',';
     if(pattern.image.empty()||pattern.clipped){++invalid;log<<",,,,invalid_crop\n";
       if(details.is_open())details<<frame.index<<','<<c.index<<','<<pattern.image.cols<<','<<pattern.image.rows<<",invalid_crop,,,,,,,,,\n";
       continue;}
     const auto input=classifier_input(pattern.image);auto tensor=request.get_input_tensor();std::copy(input.ptr<float>(),input.ptr<float>()+1024,tensor.data<float>());
     request.infer();auto output_tensor=request.get_output_tensor();auto p=classifier_probabilities(output_tensor.data<float>());
     const size_t id=std::max_element(p.begin(),p.end())-p.begin();const bool accepted=p[id]>=threshold;
     ++totals[labels[id]];if(accepted)++high[labels[id]];
     log<<id<<','<<labels[id]<<','<<p[id]<<','<<accepted<<",classified\n";
     if(details.is_open()){
       details<<frame.index<<','<<c.index<<','<<pattern.image.cols<<','<<pattern.image.rows<<",classified";
       for(double value:p)details<<','<<value;details<<'\n';
       if(evidence_frames.count(frame.index)){
         const auto stem=evidence+"/frame_"+std::to_string(frame.index)+"_"+std::to_string(c.index);
         cv::Mat pixels;input.convertTo(pixels,CV_8U,255);
         if(!cv::imwrite(stem+"_crop.png",pattern.image)||!cv::imwrite(stem+"_input.png",pixels))
           throw std::runtime_error("cannot save evidence image");
       }
     }
   }
 }
 log.flush();if(!log||processed!=expected)throw std::runtime_error("incomplete classification replay");
 if(details.is_open()){details.flush();if(!details)throw std::runtime_error("incomplete evidence log");}
 std::cout<<"candidates="<<processed<<" invalid_crop="<<invalid<<'\n';
 for(auto [name,count]:totals)std::cout<<name<<" total="<<count<<" above_threshold="<<high[name]<<'\n';
 std::cout<<"Classification is not lamp completeness, physical identity, calibrated confidence, or video accuracy. Not used for tracking.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
