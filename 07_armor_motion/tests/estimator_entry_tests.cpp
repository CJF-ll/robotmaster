#include "estimator.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace outpost;
namespace {
void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
std::string contents(const std::filesystem::path& p){std::ifstream f(p);std::stringstream s;s<<f.rdbuf();return s.str();}
int run(const std::vector<std::string>& args,const std::filesystem::path& log){
  const int fd=open(log.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);check(fd>=0,"cannot create process log");
  const auto child=fork();check(child>=0,"cannot fork entry test");
  if(child==0){dup2(fd,STDERR_FILENO);dup2(fd,STDOUT_FILENO);close(fd);
    std::vector<char*> raw;for(const auto& a:args)raw.push_back(const_cast<char*>(a.c_str()));raw.push_back(nullptr);
    execv(raw[0],raw.data());_exit(127);
  }
  close(fd);int status=0;check(waitpid(child,&status,0)==child,"cannot wait for entry test");
  return WIFEXITED(status)?WEXITSTATUS(status):128;
}
}
int main(int argc,char** argv){try{
  if(argc!=5)return 2;
  const auto image=Config::load(argv[1]);auto node=YAML::LoadFile(argv[1]);auto g=node["geometry"];
  g["mode"]="calibrated_3d";g["undistort"]=false;g["calibration_width"]=1280;g["calibration_height"]=720;
  g["camera_matrix"]=std::vector<double>{1100,0,640,0,1100,360,0,0,1};
  g["distortion"]=std::vector<double>{0,0,0,0,0};
  g["camera_from_target_rotation"]=std::vector<double>{0,1,0,0,0,-1,-1,0,0};
  g["lightbar_length_m"]=.06;g["lightbar_center_distance_m"]=.135;
  g["rotation_radius_m"]=.27;g["armor_pitch_deg"]=0;g["slot_height_offsets_m"]=std::vector<double>{0,0,0};
  node["input"]["path"]="/nonexistent/entry-test-must-reject-mode-before-opening-video.mp4";
  const auto rigid=Config::parse(YAML::Dump(node));
  auto a=make_estimator(image,{668,688}),b=make_estimator(rigid,{1280,720});
  check(std::string(a->mode())=="image_2d"&&std::string(b->mode())=="calibrated_3d","factory selected wrong mode");
  check(!a->update(0,{},.1).rigid&&b->update(0,{},.1).rigid.has_value(),"runtime result mode lost");
  bool mismatch=false;try{a->reconfigure(rigid,{1280,720});}catch(const std::exception&){mismatch=true;}
  check(mismatch,"cross-mode reconfiguration accepted");
  auto invalid=image;invalid.geometry.mode="missing";mismatch=false;
  try{make_estimator(invalid,{668,688});}catch(const std::exception&){mismatch=true;}
  check(mismatch,"factory silently fell back for unknown mode");
  char name[]="/tmp/outpost-entry-contract-XXXXXX";const auto made=mkdtemp(name);check(made,"cannot reserve test directory");
  const std::filesystem::path dir(made),config=dir/"rigid.yaml";
  {std::ofstream f(config);f<<YAML::Dump(node);check(bool(f),"cannot save test config");}
  for(int tool=2;tool<5;++tool){
    const auto output=dir/("output-"+std::to_string(tool)+".csv"),log=dir/("error-"+std::to_string(tool)+".txt");
    std::vector<std::string> args{argv[tool],config.string(),output.string()};
    if(tool==3)args.push_back((dir/"intentionally-absent-shape-settings.yaml").string());
    check(run(args,log)==1,"2D evaluation did not reject calibrated mode");
    check(contents(log).find("supports image_2d only; requested calibrated_3d")!=std::string::npos,"evaluation rejected for wrong reason");
    check(!std::filesystem::exists(output)&&!std::filesystem::exists(output.string()+".confidence.csv"),"rejected mode created evaluation output");
  }
  std::filesystem::remove_all(dir); // Exact directory created by this test only.
  std::cout<<"shared estimator factory and 3 CLI evaluation rejection contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
