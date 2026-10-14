#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
struct Segment {long first,last;std::string name;};
struct Group {int issued=0;std::map<std::string,int> status;std::vector<double> errors;};
std::vector<std::string> split(const std::string& line){std::vector<std::string> a;std::stringstream s(line);std::string x;
  while(std::getline(s,x,','))a.push_back(x);if(!line.empty()&&line.back()==',')a.push_back("");return a;}
std::map<std::string,Group> summarize(std::istream& in,const std::vector<Segment>& segments){
  std::map<std::string,Group> groups;std::string line;
  if(!std::getline(in,line)||line!="frame,target_frame,id,status,origin_x,origin_y,predicted_x,predicted_y,observed_x,observed_y,error_px,hold_error_px,shadow_error_px")
    throw std::runtime_error("wrong audit schema");
  std::set<std::string> keys;
  const std::set<std::string> states{"scored","no_accepted_target","interrupted_history","timestamp_skipped","end_of_video"};
  while(std::getline(in,line)){const auto f=split(line);if(f.size()!=13||!states.count(f[3]))throw std::runtime_error("invalid audit row");
    if(!keys.insert(f[0]+","+f[1]+","+f[2]).second)throw std::runtime_error("duplicate forecast");
    const long frame=std::stol(f[0]);std::string scenario="unlabelled";
    for(const auto& segment:segments)if(frame>=segment.first&&frame<=segment.last){scenario=segment.name;break;}
    double error=0;if(f[3]=="scored"){error=std::stod(f[10]);if(!std::isfinite(error)||error<0)throw std::runtime_error("invalid error");}
    for(const auto& name:{std::string("all"),scenario}){auto& group=groups[name];++group.issued;++group.status[f[3]];
      if(f[3]=="scored")group.errors.push_back(error);}
  }return groups;
}
int main(int argc,char** argv){try{
  if(argc==2&&std::string(argv[1])=="--self-test"){
    std::istringstream input("frame,target_frame,id,status,origin_x,origin_y,predicted_x,predicted_y,observed_x,observed_y,error_px,hold_error_px,shadow_error_px\n0,3,1,scored,0,0,0,0,1,0,1,1,\n1,4,1,no_accepted_target,0,0,0,0,,,,,\n2,5,1,end_of_video,0,0,0,0,,,,,\n");
    auto result=summarize(input,{{0,1,"start_stop"}});
    if(result["all"].issued!=3||result["start_stop"].issued!=2||result["start_stop"].errors.size()!=1||
       result["unlabelled"].status["end_of_video"]!=1)throw std::runtime_error("scenario accounting wrong");return 0;
  }
  if(argc!=3){std::cerr<<"usage: prediction_scenarios audit.csv reviewed_segments.yaml\n";return 2;}
  const auto root=YAML::LoadFile(argv[2]);
  if(root["schema_version"].as<int>()!=1||!root["reviewed_by_human"].as<bool>(false)||
     root["audit_path"].as<std::string>()!=argv[1]||!root["segments"].IsSequence())
    throw std::runtime_error("unreviewed segments or wrong audit association");
  std::vector<Segment> segments;const std::set<std::string> allowed{"front","side","start_stop","reacquisition","occlusion"};
  for(const auto& n:root["segments"]){Segment s{n["first"].as<long>(),n["last"].as<long>(),n["scenario"].as<std::string>()};
    if(s.first<0||s.last<s.first||!allowed.count(s.name))throw std::runtime_error("invalid segment");segments.push_back(s);}
  if(segments.empty())throw std::runtime_error("no reviewed segments");
  std::sort(segments.begin(),segments.end(),[](const auto&a,const auto&b){return a.first<b.first;});
  for(size_t i=1;i<segments.size();++i)if(segments[i].first<=segments[i-1].last)throw std::runtime_error("overlapping segments");
  std::ifstream in(argv[1]);if(!in)throw std::runtime_error("cannot read audit");
  for(auto& [name,g]:summarize(in,segments)){std::cout<<name<<" issued="<<g.issued;
    for(const auto& [status,count]:g.status)std::cout<<' '<<status<<'='<<count;
    if(!g.errors.empty()){std::sort(g.errors.begin(),g.errors.end());double sum=0;for(double e:g.errors)sum+=e;
      std::cout<<" mean_px="<<sum/g.errors.size()<<" p95_px="<<g.errors[size_t(std::ceil(.95*g.errors.size()))-1]<<" max_px="<<g.errors.back();}
    std::cout<<'\n';}
  std::cout<<"Segments classify issue frames only; errors remain detector-relative, not manual ground truth. All unscored and unlabelled forecasts retained.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
