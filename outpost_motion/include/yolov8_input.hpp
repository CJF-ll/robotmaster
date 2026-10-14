#pragma once
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
namespace outpost {
// Fixed model contract: BGR8 -> top-left letterbox -> RGB float NCHW.
inline cv::Size prepare_yolov8_input(const cv::Mat& image,cv::Mat& scratch,
                                    float* data,size_t capacity){
  constexpr size_t elements=3*416*416;
  if(image.empty()||image.type()!=CV_8UC3||!data||capacity!=elements)
    throw std::runtime_error("invalid YOLOv8 input/tensor buffer");
  const double scale=std::min(416./image.cols,416./image.rows);
  const cv::Size resized(std::max(1,int(image.cols*scale)),std::max(1,int(image.rows*scale)));
  scratch.create(416,416,CV_8UC3);
  scratch.setTo(cv::Scalar());
  cv::resize(image,scratch(cv::Rect(0,0,resized.width,resized.height)),resized);
  for(int y=0;y<416;++y){const auto* row=scratch.ptr<cv::Vec3b>(y);
    for(int x=0;x<416;++x){const auto p=row[x];
      for(int c=0;c<3;++c)data[c*416*416+y*416+x]=p[2-c]/255.f;}
  }
  return resized;
}
}
