#include "core.hpp"
#include "detection_backend.hpp"
#include "association.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <limits>
using namespace outpost;
struct Truth {std::array<cv::Point2f,4> p;cv::Point2f center;};
struct Stats {int frames=0,truth=0,detected=0,matched=0;double center=0,corner=0;};
int main(int argc,char** argv){try{
  if(argc!=5){std::cerr<<"usage: label_evaluation config.yaml manual_labels.yaml output.csv max_center_distance_px\n";return 2;}
  const double gate=std::stod(argv[4]);if(!std::isfinite(gate)||gate<=0)throw std::runtime_error("invalid match gate");
  if(std::filesystem::exists(argv[3]))throw std::runtime_error("output exists");
  auto config=Config::load(argv[1]);const auto root=YAML::LoadFile(argv[2]);
  if(root["schema_version"].as<int>()!=1||root["coordinate_space"].as<std::string>()!="raw"||config.geometry.undistort)
    throw std::runtime_error("this label evaluator requires raw coordinates and undistort=false");
  if(!root["reviewed_by_human"].as<bool>(false)||root["frames"].size()==0)
    throw std::runtime_error("no reviewed manual labels; template is not ground truth");
  if(root["video_path"].as<std::string>()!=config.input)throw std::runtime_error("label/video mismatch");
  const cv::Size expected(root["width"].as<int>(),root["height"].as<int>());
  std::map<long,YAML::Node> labels;
  for(const auto& f:root["frames"]){const long index=f["frame"].as<long>();
    if(index<0||!f["complete"].as<bool>(false)||!f["objects"].IsSequence()||
       !labels.emplace(index,f).second)throw std::runtime_error("incomplete/duplicate label frame");
  }
  std::ofstream out(argv[3]);if(!out)throw std::runtime_error("cannot create evaluation output");
  out<<"frame,scenario,truth_count,detection_count,matched,false_positive,false_negative,mean_center_px,mean_corner_px\n";
  auto detector=make_detector(config.detector.backend,config.detector);VideoSource source(config);
  std::map<std::string,Stats> stats;Frame frame;int visited=0,total=0;
  while(source.next(frame)){++total;const auto it=labels.find(frame.index);if(it==labels.end())continue;++visited;
    if(frame.image.size()!=expected)throw std::runtime_error("label resolution mismatch");
    const auto f=it->second;const auto scenario=f["scenario"].as<std::string>();
    const std::set<std::string> allowed{"front","side","start_stop","reacquisition","occlusion","negative"};
    if(!allowed.count(scenario))throw std::runtime_error("unknown scenario");
    std::vector<Truth> truth;std::set<std::string> ids;
    for(const auto& object:f["objects"]){Truth t{};
      if(!ids.insert(object["id"].as<std::string>()).second||object["points"].size()!=4)
        throw std::runtime_error("invalid object id/points");
      for(int k=0;k<4;++k){const auto xy=object["points"][k].as<std::vector<float>>();
        if(xy.size()!=2||!std::isfinite(xy[0])||!std::isfinite(xy[1])||xy[0]<0||xy[1]<0||
           xy[0]>=expected.width||xy[1]>=expected.height)throw std::runtime_error("invalid truth point");
        t.p[k]={xy[0],xy[1]};t.center+=t.p[k]*.25f;}
      const std::vector<cv::Point2f> poly(t.p.begin(),t.p.end());
      if(!cv::isContourConvex(poly)||cv::contourArea(poly)<=4||t.p[0].x>=t.p[1].x||t.p[3].x>=t.p[2].x||
         t.p[0].y>=t.p[3].y||t.p[1].y>=t.p[2].y)throw std::runtime_error("truth must be TL TR BR BL centerline endpoints");
      truth.push_back(t);
    }
    const auto batch=detector->detect(frame.image);const auto& detections=batch.observations;
    std::vector<std::vector<double>> cost(truth.size(),std::vector<double>(detections.size(),std::numeric_limits<double>::infinity()));
    for(size_t i=0;i<truth.size();++i)for(size_t j=0;j<detections.size();++j){const double d=cv::norm(truth[i].center-detections[j].center);
      if(d<=gate)cost[i][j]=d;}
    const auto match=associate(cost,detections.size(),gate+1,0);
    Stats s;s.frames=1;s.truth=int(truth.size());s.detected=int(detections.size());
    for(size_t i=0;i<truth.size();++i)if(match.assignment[i]>=0){++s.matched;
      const auto& d=detections[match.assignment[i]];s.center+=cv::norm(truth[i].center-d.center);
      for(int k=0;k<4;++k)s.corner+=cv::norm(truth[i].p[k]-d.measurement_points[k])*.25;}
    out<<frame.index<<','<<scenario<<','<<s.truth<<','<<s.detected<<','<<s.matched<<','<<s.detected-s.matched<<','<<s.truth-s.matched<<',';
    if(s.matched)out<<s.center/s.matched;out<<',';if(s.matched)out<<s.corner/s.matched;out<<'\n';
    for(const auto& name:{std::string("all"),scenario}){auto& sum=stats[name];sum.frames+=s.frames;
      sum.truth+=s.truth;sum.detected+=s.detected;sum.matched+=s.matched;sum.center+=s.center;sum.corner+=s.corner;}
  }
  if(visited!=int(labels.size()))throw std::runtime_error("label frame beyond video; output incomplete");
  out.flush();if(!out)throw std::runtime_error("incomplete evaluation write");
  for(const auto& [name,s]:stats){std::cout<<name<<" frames="<<s.frames<<" TP="<<s.matched<<" FP="<<s.detected-s.matched<<" FN="<<s.truth-s.matched;
    if(s.matched)std::cout<<" center_px="<<s.center/s.matched<<" corner_px="<<s.corner/s.matched;std::cout<<'\n';}
  std::cout<<"unlabelled_frames="<<total-visited<<"; matching gate="<<gate<<"px; unmatched/ambiguous predictions count as FP/FN\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
