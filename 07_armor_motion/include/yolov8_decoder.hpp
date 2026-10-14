#pragma once
#include "detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace outpost {
struct NeuralCandidate {Detection detection;float confidence;int raw_class;cv::Rect2f box;};
// Rect2f::area and intersection arithmetic can overflow even for finite inputs.
inline double neural_box_iou(const cv::Rect2f& a,const cv::Rect2f& b){
  for(const auto& box:{a,b})
    if(!std::isfinite(box.x)||!std::isfinite(box.y)||!std::isfinite(box.width)||
       !std::isfinite(box.height)||box.width<=0||box.height<=0)return 0;
  const double left=std::max(double(a.x),double(b.x)),top=std::max(double(a.y),double(b.y));
  const double right=std::min(double(a.x)+a.width,double(b.x)+b.width);
  const double bottom=std::min(double(a.y)+a.height,double(b.y)+b.height);
  const double intersection=std::max(0.,right-left)*std::max(0.,bottom-top);
  const double united=double(a.width)*a.height+double(b.width)*b.height-intersection;
  return united>0?std::clamp(intersection/united,0.,1.):0;
}
// Reference YOLOv8 IR contract: [1,14,N] = xywh,2 scores,4 xy keypoints.
// Classes are kept raw; no digit identity or enemy-color mapping is invented.
inline std::vector<NeuralCandidate> decode_yolov8(const float* data,size_t channels,size_t n,
    cv::Size image,cv::Size resized,double score_threshold,double nms_threshold){
  if(!data||channels!=14||n==0||image.width<=0||image.height<=0||resized.width<=0||resized.height<=0||
     !std::isfinite(score_threshold)||score_threshold<0||score_threshold>1||
     !std::isfinite(nms_threshold)||nms_threshold<0||nms_threshold>1)
    throw std::runtime_error("invalid YOLOv8 tensor/decoder configuration");
  const double sx=double(resized.width)/image.width,sy=double(resized.height)/image.height;
  std::vector<NeuralCandidate> candidates;
  for(size_t i=0;i<n;++i){auto v=[&](int channel){return data[size_t(channel)*n+i];};
    bool finite=true;for(int k=0;k<14;++k)finite=finite&&std::isfinite(v(k));if(!finite)continue;
    const int cls=v(5)>v(4)?1:0;const float score=v(4+cls);
    if(score<score_threshold||score>1||v(2)<=0||v(3)<=0)continue;
    NeuralCandidate c{};c.confidence=score;c.raw_class=cls;
    c.detection.kind=ObservationKind::NeuralKeypoints;
    c.detection.detection_confidence=score;c.detection.raw_network_class=cls;
    c.box={float((v(0)-v(2)/2)/sx),float((v(1)-v(3)/2)/sy),float(v(2)/sx),float(v(3)/sy)};
    if(!std::isfinite(c.box.x)||!std::isfinite(c.box.y)||!std::isfinite(c.box.width)||
       !std::isfinite(c.box.height)||c.box.width<=0||c.box.height<=0)continue;
    std::array<cv::Point2f,4> p;
    for(int k=0;k<4;++k){p[k]={float(v(6+2*k)/sx),float(v(7+2*k)/sy)};
      if(!std::isfinite(p[k].x)||!std::isfinite(p[k].y)||p[k].x<0||p[k].x>=image.width||p[k].y<0||p[k].y>=image.height)finite=false;}
    if(!finite)continue;
    std::sort(p.begin(),p.end(),[](auto a,auto b){return a.y==b.y?a.x<b.x:a.y<b.y;});
    if(p[0].x>p[1].x)std::swap(p[0],p[1]);
    if(p[2].x<p[3].x)std::swap(p[2],p[3]);
    std::vector<cv::Point2f> poly(p.begin(),p.end());
    if(!cv::isContourConvex(poly)||cv::contourArea(poly)<=4||p[0].x>=p[1].x||p[3].x>=p[2].x||
       p[0].y>=p[3].y||p[1].y>=p[2].y)continue;
    c.detection.measurement_points=p;c.detection.corners=p;
    for(auto point:p)c.detection.center+=point*.25f;
    candidates.push_back(c);
  }
  std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.confidence>b.confidence;});
  std::vector<NeuralCandidate> keep;
  for(const auto& c:candidates){bool suppressed=false;for(const auto& old:keep){
      if(neural_box_iou(c.box,old.box)>nms_threshold){suppressed=true;break;}}
    if(!suppressed)keep.push_back(c);
  }return keep;
}
}
