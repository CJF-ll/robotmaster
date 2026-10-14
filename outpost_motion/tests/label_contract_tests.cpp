#include <yaml-cpp/yaml.h>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <unistd.h>
std::string quote(const std::string& s){std::string q="'";for(char c:s)q+=c=='\''?"'\\''":std::string(1,c);return q+"'";}
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv){try{
  if(argc!=3)return 2;char name[]="/tmp/outpost-label-test-XXXXXX";
  const auto dir=mkdtemp(name);check(dir,"cannot create fixture directory");
  const std::filesystem::path root(dir);
  struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove_all(path,e);}} cleanup{root};
  const auto video=(root/"input.avi").string();
  cv::VideoWriter writer(video,cv::VideoWriter::fourcc('F','F','V','1'),30,{160,120});
  check(writer.isOpened(),"lossless fixture writer unavailable");
  cv::Mat frame(120,160,CV_8UC3,cv::Scalar());
  for(int x:{30,60})cv::rectangle(frame,{x,40,4,20},{0,80,255},cv::FILLED);
  writer.write(frame);writer.write(cv::Mat::zeros(frame.size(),frame.type()));writer.release();
  auto cfg=YAML::LoadFile(argv[1]);cfg["input"]["path"]=video;cfg["geometry"]["undistort"]=false;
  {std::ofstream f(root/"config.yaml");f<<cfg;}
  YAML::Node labels;labels["schema_version"]=1;labels["video_path"]=video;labels["coordinate_space"]="raw";
  labels["width"]=160;labels["height"]=120;labels["reviewed_by_human"]=true;
  // Synthetic fixtures are labelled analytically; this flag only exercises parser validation.
  YAML::Node f;f["frame"]=0;f["scenario"]="front";f["complete"]=true;
  YAML::Node obj;obj["id"]="synthetic";
  for(auto p:std::vector<std::vector<double>>{{31.5,40},{61.5,40},{61.5,59},{31.5,59}})obj["points"].push_back(p);
  f["objects"].push_back(obj);labels["frames"].push_back(f);
  YAML::Node empty;empty["frame"]=1;empty["scenario"]="negative";empty["complete"]=true;
  empty["objects"]=YAML::Node(YAML::NodeType::Sequence);labels["frames"].push_back(empty);
  int run=0;auto evaluate=[&](){const auto file=root/("labels"+std::to_string(run)+".yaml");
    const auto output=root/("output"+std::to_string(run++)+".csv");{std::ofstream stream(file);stream<<labels;}
    const auto command=quote(argv[2])+" "+quote((root/"config.yaml").string())+" "+quote(file.string())+" "+quote(output.string())+" 30";
    return std::make_pair(std::system(command.c_str()),output);};
  auto result=evaluate();check(result.first==0,"valid fixture rejected");
  {std::ifstream file(result.second);std::string line;std::getline(file,line);std::getline(file,line);
    check(line.find("0,front,1,1,1,0,0,")==0,"matching accounting wrong");}
  labels["frames"][0]["objects"]=YAML::Node(YAML::NodeType::Sequence);
  result=evaluate();check(result.first==0,"negative label rejected");
  {std::ifstream file(result.second);std::string line;std::getline(file,line);std::getline(file,line);
    check(line=="0,front,0,1,0,1,0,,","false positive omitted");}
  labels["frames"][1]["objects"].push_back(obj);
  result=evaluate();check(result.first==0,"missing detection fixture rejected");
  {std::ifstream file(result.second);std::string line;std::getline(file,line);std::getline(file,line);std::getline(file,line);
    check(line=="1,negative,1,0,0,0,1,,","false negative omitted");}
  labels["reviewed_by_human"]=false;check(evaluate().first!=0,"unreviewed labels accepted");
  labels["reviewed_by_human"]=true;labels["frames"][0]["complete"]=false;
  check(evaluate().first!=0,"incomplete frame accepted");
  std::cout<<"manual-label evaluator contracts passed on synthetic fixtures\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
