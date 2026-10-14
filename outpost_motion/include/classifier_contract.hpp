#pragma once
#include <opencv2/imgproc.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace outpost {
struct Pattern {cv::Mat image;bool clipped=false;};
inline Pattern classifier_pattern(const cv::Mat& image,const std::array<cv::Point2f,4>& p,double half_scale){
  if(image.empty()||image.type()!=CV_8UC3||!std::isfinite(half_scale)||half_scale<=0)
    throw std::runtime_error("invalid classifier input/scale");
  for(auto q:p)if(!std::isfinite(q.x)||!std::isfinite(q.y))throw std::runtime_error("nonfinite pattern point");
  if(!cv::isContourConvex(std::vector<cv::Point2f>(p.begin(),p.end()))||p[0].x>=p[1].x||p[3].x>=p[2].x||p[0].y>=p[3].y||p[1].y>=p[2].y)return {};
  const auto lt=(p[0]+p[3])*.5f-(p[3]-p[0])*float(half_scale);
  const auto lb=(p[0]+p[3])*.5f+(p[3]-p[0])*float(half_scale);
  const auto rt=(p[1]+p[2])*.5f-(p[2]-p[1])*float(half_scale);
  const auto rb=(p[1]+p[2])*.5f+(p[2]-p[1])*float(half_scale);
  const double l=std::min(lt.x,lb.x),r=std::max(rt.x,rb.x),t=std::min(lt.y,rt.y),b=std::max(lb.y,rb.y);
  for(double v:{l,r,t,b})if(!std::isfinite(v))return {};
  Pattern result;result.clipped=l<0||t<0||r>image.cols||b>image.rows;
  const int x=int(std::clamp(l,0.,double(image.cols))),y=int(std::clamp(t,0.,double(image.rows)));
  const int right=int(std::clamp(r,0.,double(image.cols))),bottom=int(std::clamp(b,0.,double(image.rows)));
  if(right>x&&bottom>y){
    const int width=right-x,height=bottom-y;
    const double scale=std::min(32./width,32./height);
    // Reject an unusable candidate before either inference caller sees it.
    // Do not stretch a zero-sized axis into invented classifier pixels.
    if(int(width*scale)>0&&int(height*scale)>0)
      result.image=image(cv::Rect(x,y,width,height));
  }
  return result;
}
inline cv::Mat classifier_input(const cv::Mat& pattern){
  if(pattern.empty()||pattern.type()!=CV_8UC3)throw std::runtime_error("empty/non-BGR pattern");
  cv::Mat gray,input(32,32,CV_8UC1,cv::Scalar());cv::cvtColor(pattern,gray,cv::COLOR_BGR2GRAY);
  const double scale=std::min(32./gray.cols,32./gray.rows);
  const int w=int(gray.cols*scale),h=int(gray.rows*scale);
  if(w<=0||h<=0)throw std::runtime_error("pattern aspect ratio unsupported");
  cv::resize(gray,input(cv::Rect(0,0,w,h)),{w,h});input.convertTo(input,CV_32F,1./255);return input;
}
inline std::array<double,9> classifier_probabilities(const float* values){
  if(!values)throw std::runtime_error("missing logits");
  double maximum=-INFINITY;
  for(int i=0;i<9;++i){if(!std::isfinite(values[i]))throw std::runtime_error("nonfinite logit");maximum=std::max(maximum,double(values[i]));}
  std::array<double,9> result{};double sum=0;for(int i=0;i<9;++i){result[i]=std::exp(values[i]-maximum);sum+=result[i];}
  for(auto& value:result)value/=sum;
  return result;
}
}
