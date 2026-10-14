#include "core.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>
#include <limits>
using namespace outpost;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F action){bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}
  require(rejected,"invalid preprocessing accepted");}
int main(int argc,char** argv){
  if(argc!=2)return 2;
  const cv::Size size(80,60);
  cv::Mat image(size,CV_8UC3);cv::randu(image,0,255);
  GeometryConfig g;
  ImagePreprocessor raw(g,size);
  require(!raw.rectified()&&cv::norm(image,raw.process(image),cv::NORM_INF)==0,"raw image changed");
  g.undistort=true;rejects([&]{ImagePreprocessor invalid(g,size);});
  g.camera_matrix={70,0,39.5,0,70,29.5,0,0,1};g.distortion={0,0,0,0,0};
  g.calibration_width=80;g.calibration_height=60;
  ImagePreprocessor identity(g,size);
  require(identity.rectified()&&cv::norm(image,identity.process(image),cv::NORM_INF)==0,"zero distortion not identity");
  rejects([&]{identity.process(cv::Mat(30,40,CV_8UC3));});
  rejects([&]{ImagePreprocessor invalid(g,{40,30});});
  g.allow_uniform_resize=true;
  ImagePreprocessor resized(g,{40,30});
  require(resized.camera_matrix().at<double>(0,0)==35&&resized.camera_matrix().at<double>(0,2)==19.75,
          "calibration scaling incorrect");
  rejects([&]{ImagePreprocessor invalid(g,{40,31});});
  // Coordinate ramps expose the inverse remap; independent radial formula.
  cv::Mat ramp(size,CV_32FC2);
  for(int y=0;y<size.height;++y)for(int x=0;x<size.width;++x)ramp.at<cv::Vec2f>(y,x)={float(x),float(y)};
  g.distortion[0]=.2;
  ImagePreprocessor radial(g,size);const auto result=radial.process(ramp);
  for(int y=10;y<50;y+=7)for(int x=10;x<70;x+=9){
    const double nx=(x-39.5)/70,ny=(y-29.5)/70,f=1+.2*(nx*nx+ny*ny);
    const auto p=result.at<cv::Vec2f>(y,x);
    require(std::abs(p[0]-(70*nx*f+39.5))<.017&&std::abs(p[1]-(70*ny*f+29.5))<.017,"radial mapping incorrect");
  }
  auto invalid=g;invalid.camera_matrix[1]=1;rejects([&]{ImagePreprocessor bad(invalid,size);});
  invalid=g;invalid.distortion[0]=std::numeric_limits<double>::quiet_NaN();
  rejects([&]{ImagePreprocessor bad(invalid,size);});
  ConfigRuntime runtime(argv[1]);runtime.bind_input_size(size);std::string message;
  auto root=YAML::LoadFile(argv[1]);root["geometry"]["undistort"]=true;
  require(runtime.apply_text(YAML::Dump(root),message)==ReloadResult::Rejected,"missing calibration enabled");
  auto geo=root["geometry"];
  geo["camera_matrix"]=g.camera_matrix;geo["distortion"]=g.distortion;
  geo["calibration_width"]=80;geo["calibration_height"]=60;
  require(runtime.apply_text(YAML::Dump(root),message)==ReloadResult::ResetRequired,"geometry did not reset state");
  const auto signature=runtime.current().estimation_signature;
  const auto previous=runtime.preprocessor().process(image);
  geo["calibration_width"]=160;geo["calibration_height"]=120;
  require(runtime.apply_text(YAML::Dump(root),message)==ReloadResult::Rejected,"size mismatch committed");
  require(runtime.current().estimation_signature==signature&&
          cv::norm(previous,runtime.preprocessor().process(image),cv::NORM_INF)==0,"failed update changed pipeline");
  geo["allow_uniform_resize"]=true;
  require(runtime.apply_text(YAML::Dump(root),message)==ReloadResult::ResetRequired,"explicit resizing not applied");
  geo["undistort"]=false;
  require(runtime.apply_text(YAML::Dump(root),message)==ReloadResult::ResetRequired,"disable did not reset state");
  require(!runtime.preprocessor().rectified()&&cv::norm(image,runtime.preprocessor().process(image),cv::NORM_INF)==0,
          "disable did not restore raw image");
  rejects([&]{runtime.bind_input_size({40,30});});
  std::cout<<"preprocessing calibration, mapping and atomic reload contracts passed\n";
}
