#include "core.hpp"
#include "calibrated_video.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
using namespace outpost;
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
std::vector<std::string> columns(const std::string& line){std::stringstream s(line);std::string p;std::vector<std::string> result;
  while(std::getline(s,p,','))result.push_back(p);return result;}
int main(int argc,char** argv){try{
  if(argc!=3)return 2;char name[]="/tmp/outpost-calibrated-test-XXXXXX";
  char* created=mkdtemp(name);require(created,"cannot reserve synthetic test directory");
  const std::filesystem::path dir(created);auto node=YAML::LoadFile(argv[1]);auto g=node["geometry"];
  const cv::Mat k=(cv::Mat_<double>(3,3)<<1100,0,640,0,1100,360,0,0,1);
  const cv::Matx33d basis(0,1,0,0,0,-1,-1,0,0);
  g["mode"]="calibrated_3d";g["undistort"]=false;g["calibration_width"]=1280;g["calibration_height"]=720;
  g["camera_matrix"]=std::vector<double>{1100,0,640,0,1100,360,0,0,1};
  const std::vector<double> distortion{-.08,.015,.001,-.001,0};
  g["distortion"]=distortion;
  g["camera_from_target_rotation"]=std::vector<double>{0,1,0,0,0,-1,-1,0,0};
  g["lightbar_length_m"]=.06;g["lightbar_center_distance_m"]=.135;
  g["rotation_radius_m"]=.27;g["armor_pitch_deg"]=0;g["slot_height_offsets_m"]=std::vector<double>{0,0,0};
  node["input"]["path"]=(dir/"synthetic.avi").string();node["display"]["enabled"]=false;
  // Render motion and raw-camera distortion independently of estimator helpers.
  cv::VideoWriter writer((dir/"synthetic.avi").string(),cv::VideoWriter::fourcc('F','F','V','1'),30,{1280,720});
  require(writer.isOpened(),"synthetic lossless encoder unavailable");
  for(int frame=0;frame<56;++frame){cv::Mat image(720,1280,CV_8UC3,cv::Scalar());
    const double angle=.50+.12*frame/30.;
    const cv::Vec3d center(-2.5,.015*frame/30.,.2),radial(std::cos(angle),std::sin(angle),0),
      tangent(-std::sin(angle),std::cos(angle),0),up(0,0,1);
    std::vector<cv::Point3d> points;
    for(int lamp:{-1,1})for(auto sign:std::array<cv::Point2d,4>{{{-1,1},{1,1},{1,-1},{-1,-1}}}){
      const auto p=basis*(center+.27*radial+(lamp*.135*.5+sign.x*.004)*tangent+sign.y*.06*.5*up);
      points.emplace_back(p[0],p[1],p[2]);}
    std::vector<cv::Point2d> projected;
    cv::projectPoints(points,cv::Vec3d(),cv::Vec3d(),k,cv::Mat(distortion),projected);
    if(frame<15||frame>=29)for(int lamp=0;lamp<2;++lamp){std::vector<cv::Point> q;
      for(int corner=0;corner<4;++corner)q.emplace_back(cvRound(projected[lamp*4+corner].x),cvRound(projected[lamp*4+corner].y));
      cv::fillConvexPoly(image,q,{0,40,255});}
    writer.write(image);
  }writer.release();
  {std::ofstream cfg(dir/"test.yaml");cfg<<YAML::Dump(node);require(bool(cfg),"cannot save synthetic config");}
  const auto executable=std::filesystem::absolute(argv[2]).string();
  const auto child=fork();require(child>=0,"cannot start actual video CLI");
  if(child==0){
    const int fd=open((dir/"process.log").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd>=0){dup2(fd,STDOUT_FILENO);dup2(fd,STDERR_FILENO);close(fd);}
    execl(executable.c_str(),executable.c_str(),(dir/"test.yaml").c_str(),"--headless","--no-video",
      "--output-dir",(dir/"run").c_str(),static_cast<char*>(nullptr));_exit(127);
  }
  int status_code=0;require(waitpid(child,&status_code,0)==child,"cannot wait for calibrated CLI");
  require(WIFEXITED(status_code)&&WEXITSTATUS(status_code)==0,"calibrated CLI execution failed; fixture retained");
  std::ifstream log(dir/"run/rigid_state.csv");std::string line;std::getline(log,line);int rows=0,valid=0,largest_epoch=0;
  const auto field_count=columns(line).size();require(field_count>=23,"rigid state schema missing required fields");
  while(std::getline(log,line)){const auto fields=columns(line);require(fields.size()==field_count,"rigid state schema changed");
    ++rows;valid+=std::stoi(fields[4]);largest_epoch=std::max(largest_epoch,std::stoi(fields[2]));}
  require(rows==56&&valid>=10,"image-to-rigid-state pipeline did not produce confirmed samples");
  require(largest_epoch>=1,"expired relative IDs reused an acquisition epoch");
  // Score actual CLI forecasts against independently generated future geometry,
  // not against the detector's next frame or the estimator's own projector.
  std::ifstream slots(dir/"run/rigid_slots.csv");std::getline(slots,line);
  int forecast_corners=0;double squared=0,maximum=0;
  while(std::getline(slots,line)){const auto f=columns(line);require(f.size()>=24,"rigid slot schema incomplete");
    if(!std::stoi(f[6]))continue;
    const double t=std::stod(f[1])+node["prediction"]["horizon_seconds"].as<double>();
    const double angle=.5+.12*t+std::stoi(f[3])*2*CV_PI/3;
    const cv::Vec3d center(-2.5,.015*t,.2),radial(std::cos(angle),std::sin(angle),0),
      tangent(-std::sin(angle),std::cos(angle),0),up(0,0,1);
    std::vector<cv::Point3d> object;
    for(auto sign:std::array<cv::Point2d,4>{{{-1,1},{1,1},{1,-1},{-1,-1}}}){
      const auto p=basis*(center+.27*radial+sign.x*.0675*tangent+sign.y*.03*up);
      object.emplace_back(p[0],p[1],p[2]);}
    std::vector<cv::Point2d> truth;cv::projectPoints(object,cv::Vec3d(),cv::Vec3d(),k,cv::Mat(distortion),truth);
    for(int corner=0;corner<4;++corner){const double error=cv::norm(truth[corner]-
      cv::Point2d(std::stod(f[16+2*corner]),std::stod(f[17+2*corner])));
      require(std::isfinite(error),"nonfinite future geometry");++forecast_corners;squared+=error*error;maximum=std::max(maximum,error);}
  }
  require(forecast_corners>=120,"calibrated CLI forecast coverage insufficient");
  const double rms=std::sqrt(squared/forecast_corners);
  std::cout<<"independent future corners="<<forecast_corners<<" rms_px="<<rms<<" max_px="<<maximum<<'\n';
  require(rms<10&&maximum<30,"calibrated CLI future corners departed from synthetic truth");
  std::ifstream status(dir/"run/run_status.txt");std::stringstream status_text;status_text<<status.rdbuf();
  require(status_text.str().find("state=completed")!=std::string::npos,"synthetic video run incomplete");
  require(!std::filesystem::exists(dir/"run/tracking-preview.mp4"),"no-video generated an MP4");
  for(const auto* file:{"timing.csv","detector_timing.csv","detection.csv","config_events.csv",
      "observation_sources.csv","backend_audit.csv","coordinates.csv","estimator_mode.txt"})
    require(std::filesystem::exists(dir/"run"/file),"shared runner omitted diagnostics in calibrated mode");
  std::cout<<"synthetic image->detection->pose->rigid pipeline frames="<<rows<<" valid="<<valid<<" epoch="<<largest_epoch<<'\n';
  std::filesystem::remove_all(dir); // Only this mkdtemp-owned synthetic fixture.
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
