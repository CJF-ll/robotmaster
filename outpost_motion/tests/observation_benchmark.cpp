#include "core.hpp"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <stdexcept>
using namespace outpost;
struct LampTruth {
  cv::Point2d center,axis;double length,width;
  cv::Point2f endpoint(int sign)const{return cv::Point2f(center+sign*.5*length*axis);}
  bool inside(cv::Point2d p)const{const auto d=p-center;
    return std::abs(d.dot(axis))<=length/2&&std::abs(d.dot({axis.y,-axis.x}))<=width/2;}
};
// Integrate an analytic rectangle over each pixel's area. Ground truth never
// comes from minAreaRect, contours, thresholding or the measured silhouette.
cv::Mat render(const LampTruth& left,const LampTruth& right,double blur){
  cv::Mat image(96,128,CV_8UC3,cv::Scalar(5,5,5));
  constexpr int samples=8;
  for(int y=0;y<image.rows;++y)for(int x=0;x<image.cols;++x){
    int covered=0;
    for(int j=0;j<samples;++j)for(int i=0;i<samples;++i){
      const cv::Point2d p(x+(i+.5)/samples-.5,y+(j+.5)/samples-.5);
      covered+=left.inside(p)||right.inside(p);
    }
    const double a=double(covered)/(samples*samples);
    image.at<cv::Vec3b>(y,x)={cv::saturate_cast<uchar>(5+5*a),
      cv::saturate_cast<uchar>(5+75*a),cv::saturate_cast<uchar>(5+235*a)};
  }
  if(blur>0)cv::GaussianBlur(image,image,cv::Size(),blur,blur,cv::BORDER_REPLICATE);
  return image;
}
struct Stats {int total=0,detected=0;double center=0,endpoints=0,max_center=0,max_endpoint=0;
  void add(double c,double e){++detected;center+=c;endpoints+=e;max_center=std::max(max_center,c);max_endpoint=std::max(max_endpoint,e);}
  void print(const char* name)const{std::cout<<name<<" total="<<total<<" detected="<<detected<<" missed="<<total-detected;
    if(detected)std::cout<<" center_mean="<<center/detected<<" endpoint_mean="<<endpoints/detected
                        <<" center_max="<<max_center<<" endpoint_case_mean_max="<<max_endpoint;
    std::cout<<'\n';}
};
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(int argc,char** argv){try{
  if(argc!=3){std::cerr<<"usage: observation_benchmark config.yaml output.csv|--self-test\n";return 2;}
  const auto config=Config::load(argv[1]);const bool test=std::string(argv[2])=="--self-test";
  std::ofstream log;
  if(!test){if(std::filesystem::exists(argv[2]))throw std::runtime_error("output exists");
    log.open(argv[2]);if(!log)throw std::runtime_error("cannot write benchmark");
    log<<std::setprecision(12)<<"case_id,tilt_deg,length,width,subpixel_phase,blur_sigma,detection_count,center_error_px,mean_endpoint_error_px\n";}
  LampTruth sanity{{30,40},{0,1},20,4};
  require(cv::norm(sanity.endpoint(-1)-cv::Point2f(30,30))<1e-6&&
          sanity.inside({31.9,49.9})&&!sanity.inside({32.1,40}),"analytic truth contract failed");
  Stats all,sharp,blurred;int index=0;
  for(double tilt:{-25.,0.,25.})for(double length:{12.,22.})for(double width:{2.,4.})
    for(double phase:{0.,.25,.5,.75})for(double sigma:{0.,1.}){
      const double angle=tilt*CV_PI/180,other=(tilt+4)*CV_PI/180;
      LampTruth left{{40+phase,44+phase},{std::sin(angle),std::cos(angle)},length,width};
      LampTruth right{{75+phase,47+phase},{std::sin(other),std::cos(other)},length*1.2,width};
      const auto image=render(left,right,sigma);Detector detector(config.detector);
      if(test&&tilt==0&&width==2&&phase==0&&sigma==0){
        int surviving_columns=0;
        for(int x=37;x<=43;++x){const auto p=image.at<cv::Vec3b>(44,x);
          surviving_columns+=p[2]>=config.detector.brightness&&int(p[2])-p[0]>=config.detector.red_difference;}
        // Under the shipped threshold, half-covered edge pixels disappear.
        // Preserve this explicit pixel-aliasing regression, not a fake success.
        if(config.detector.brightness==130&&config.detector.red_difference==35)
          require(surviving_columns==1,"thin-lamp aliasing fixture changed");
      }
      const auto found=detector.detect(image);++all.total;auto& subgroup=sigma==0?sharp:blurred;++subgroup.total;
      if(!test)log<<index<<','<<tilt<<','<<length<<','<<width<<','<<phase<<','<<sigma<<','<<found.size();
      if(found.size()==1){const auto& d=found.front();
        const cv::Point2f center((left.center+right.center)*.5);
        const std::array<cv::Point2f,4> endpoints={left.endpoint(-1),right.endpoint(-1),right.endpoint(1),left.endpoint(1)};
        double error=0;for(int k=0;k<4;++k)error+=cv::norm(d.measurement_points[k]-endpoints[k])/4;
        const double center_error=cv::norm(d.center-center);
        require(std::isfinite(center_error)&&std::isfinite(error),"nonfinite geometry metric");
        all.add(center_error,error);subgroup.add(center_error,error);
        if(!test)log<<','<<center_error<<','<<error;
        // Independent absolute contracts only on unblurred broad lamps.
        if(test&&sigma==0&&width==4){require(center_error<1.5,"clean center localization regressed");
          require(error<2,"clean endpoint localization regressed");}
      }else {if(!test)log<<",,";
        if(test&&sigma==0&&width==4)throw std::runtime_error("clean broad lamp pair missed");}
      if(!test)log<<'\n';++index;
    }
  require(all.total==96,"incomplete benchmark sweep");
  if(!test){log.flush();if(!log)throw std::runtime_error("incomplete benchmark output");}
  std::cout<<std::fixed<<std::setprecision(5);all.print("all");sharp.print("sharp");blurred.print("blurred");
  std::cout<<"Known synthetic rectangle geometry only, not labels or an accuracy claim for the supplied video.\n"
             "Missing/ambiguous detections remain in denominator; errors use detected cases only.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
