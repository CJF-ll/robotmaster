#include "core.hpp"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
using namespace outpost;
struct Emitter {
  cv::Point2d center,axis;double length,width,red;
  bool inside(cv::Point2d p)const{auto d=p-center;
    return std::abs(d.dot(axis))<=length/2&&std::abs(d.dot({axis.y,-axis.x}))<=width/2;}
  cv::Point2f end(int sign)const{return cv::Point2f(center+axis*(sign*length/2));}
};
// Independent analytic pixel integration; neither truth nor occlusion comes
// from detector contours. Synthetic red emitters do not model real reflections.
cv::Mat render(const std::vector<Emitter>& emitters,double blur){
  cv::Mat image(96,128,CV_8UC3,cv::Scalar(5,5,5));
  for(int y=0;y<image.rows;++y)for(int x=0;x<image.cols;++x){
    double red=0,green=0,blue=0;
    for(int j=0;j<8;++j)for(int i=0;i<8;++i){
      const cv::Point2d p(x+(i+.5)/8-.5,y+(j+.5)/8-.5);
      double r=5;for(const auto& e:emitters)if(e.inside(p))r=std::max(r,e.red);
      red+=r;green+=5+(r-5)*.32;blue+=5+(r-5)*.02;
    }
    image.at<cv::Vec3b>(y,x)={cv::saturate_cast<uchar>(blue/64),cv::saturate_cast<uchar>(green/64),cv::saturate_cast<uchar>(red/64)};
  }
  if(blur>0)cv::GaussianBlur(image,image,{},blur);
  return image;
}
struct Summary {int cases=0,accepted_cases=0,candidates=0,scored=0;double center=0,endpoints=0;};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char** argv){try{
  if(argc!=3){std::cerr<<"usage: weak_lamp_benchmark config.yaml output.csv|--self-test\n";return 2;}
  const auto config=Config::load(argv[1]);const bool test=std::string(argv[2])=="--self-test";
  std::ofstream log;if(!test){require(!std::filesystem::exists(argv[2]),"output exists");log.open(argv[2]);require(bool(log),"cannot open output");
    log<<std::setprecision(10)<<"case,scene,dim_red,phase,blur,candidates,full_geometry_scored,center_error,endpoint_error\n";}
  std::map<std::string,Summary> stats;int index=0;
  for(const std::string scene:{"complete","single","clipped","split","lookalike_pair","blank"})
    for(double red:{70.,100.,120.,160.,240.})for(double phase:{0.,.5})for(double blur:{0.,1.}){
      Emitter left{{40+phase,44+phase},{0,1},22,4,red};
      Emitter right{{75+phase,47+phase},{0,1},22,4,240};
      std::vector<Emitter> lights;
      if(scene=="complete"||scene=="lookalike_pair")lights={left,right};
      if(scene=="single")lights={right};
      if(scene=="clipped"){
        auto visible=left;visible.length*=.65;visible.center+=left.axis*((left.length-visible.length)/2);lights={visible,right};
      }
      if(scene=="split"){
        auto upper=left,lower=left;upper.length=lower.length=8;
        upper.center-=left.axis*7;lower.center+=left.axis*7;lights={upper,lower,right};
      }
      const auto image=render(lights,blur);Detector detector(config.detector);const auto found=detector.detect(image);
      auto& s=stats[scene];++s.cases;s.accepted_cases+=!found.empty();s.candidates+=int(found.size());
      const bool score=scene=="complete"&&found.size()==1;double ce=0,ee=0;
      if(score){const std::array<cv::Point2f,4> truth{left.end(-1),right.end(-1),right.end(1),left.end(1)};
        ce=cv::norm(found[0].center-cv::Point2f((left.center+right.center)/2));
        for(int k=0;k<4;++k)ee+=cv::norm(found[0].measurement_points[k]-truth[k])/4;
        ++s.scored;s.center+=ce;s.endpoints+=ee;
      }
      if(test){
        if(scene=="blank"||scene=="single")require(found.empty(),"blank/single lamp formed a full observation");
        if(scene=="complete"&&red==240&&blur==0){require(score,"bright clean pair missed");require(ce<1&&ee<2,"bright geometry contract failed");}
        // A semantic non-target with identical pixels is necessarily indistinguishable
        // to a geometric detector. Do not turn this into a fake negative-test pass.
        if(scene=="lookalike_pair"&&red==240&&blur==0)require(found.size()==1,"identical-pixel fixture changed");
      }
      if(!test){log<<index<<','<<scene<<','<<red<<','<<phase<<','<<blur<<','<<found.size()<<','<<score<<',';
        if(score)log<<ce<<','<<ee;else log<<',';log<<'\n';}
      ++index;
    }
  require(index==120,"incomplete benchmark");
  for(const auto& [scene,s]:stats){std::cout<<scene<<" cases="<<s.cases<<" accepted_cases="<<s.accepted_cases<<" candidates="<<s.candidates<<" scored="<<s.scored;
    if(s.scored)std::cout<<" center_mean="<<s.center/s.scored<<" endpoint_mean="<<s.endpoints/s.scored;std::cout<<'\n';}
  if(!test){log.flush();require(bool(log),"incomplete output");}
  std::cout<<"Synthetic stress cases, not video ground truth. Clipped/split detections are counted, not scored as complete geometry.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
