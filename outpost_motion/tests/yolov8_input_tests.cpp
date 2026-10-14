#include "yolov8_input.hpp"
#include <vector>
#include <iostream>
using namespace outpost;
void check(bool value){if(!value)throw std::runtime_error("YOLO input contract failed");}
int main(){try{
  constexpr size_t count=3*416*416;
  std::vector<float> guarded(count+2,-7);cv::Mat scratch;
  auto verify=[&](const cv::Mat& image){
    const auto before=image.clone();
    const auto resized=prepare_yolov8_input(image,scratch,guarded.data()+1,count);
    const double scale=std::min(416./image.cols,416./image.rows);
    check(resized==cv::Size(std::max(1,int(image.cols*scale)),std::max(1,int(image.rows*scale))));
    cv::Mat expected(416,416,CV_8UC3,cv::Scalar());
    cv::resize(image,expected(cv::Rect(0,0,resized.width,resized.height)),resized);
    for(int y=0;y<416;++y)for(int x=0;x<416;++x)for(int c=0;c<3;++c)
      check(guarded[1+c*416*416+y*416+x]==expected.at<cv::Vec3b>(y,x)[2-c]/255.f);
    check(guarded.front()==-7&&guarded.back()==-7);
    check(cv::norm(image,before,cv::NORM_INF)==0);
  };
  // Alternating aspect ratios must clear both possible padding regions.
  verify(cv::Mat(416,416,CV_8UC3,cv::Scalar(13,79,255)));
  const auto* buffer=scratch.data;
  verify(cv::Mat(50,100,CV_8UC3,cv::Scalar(255,0,17)));
  verify(cv::Mat(100,50,CV_8UC3,cv::Scalar(0,255,31)));
  check(buffer==scratch.data);
  cv::Mat parent(77,109,CV_8UC3);cv::RNG rng(42);rng.fill(parent,cv::RNG::UNIFORM,0,256);
  auto roi=parent(cv::Rect(3,5,89,61));check(!roi.isContinuous());verify(roi);
  verify(cv::Mat(1,1,CV_8UC3,cv::Scalar(11,22,33)));
  verify(cv::Mat(1,1000,CV_8UC3,cv::Scalar(11,22,33)));
  for(int kind=0;kind<4;++kind){
    const auto saved=guarded;bool rejected=false;
    try{prepare_yolov8_input(kind==0?cv::Mat():cv::Mat(2,2,kind==1?CV_8UC1:CV_8UC3),
        scratch,kind==2?nullptr:guarded.data()+1,kind==3?count-1:count);}catch(const std::runtime_error&){rejected=true;}
    check(rejected&&saved==guarded);
  }
  std::cout<<"YOLO preprocessing layout, padding, stride and buffer contracts passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
