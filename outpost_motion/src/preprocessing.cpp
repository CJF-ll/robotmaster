#include "core.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>

namespace outpost {
ImagePreprocessor::ImagePreprocessor(const GeometryConfig& g,cv::Size size):size_(size){
  if(size.width<=0||size.height<=0)throw std::runtime_error("invalid input resolution");
  if(!g.undistort)return;
  if(g.camera_matrix.size()!=9||g.distortion.empty()||g.calibration_width<=0||g.calibration_height<=0)
    throw std::runtime_error("missing undistortion calibration");
  for(const auto* values:{&g.camera_matrix,&g.distortion})
    for(double v:*values)if(!std::isfinite(v))throw std::runtime_error("nonfinite calibration");
  const auto& k=g.camera_matrix;
  if(k[0]<=0||k[4]<=0||k[1]!=0||k[3]!=0||k[6]!=0||k[7]!=0||k[8]!=1)
    throw std::runtime_error("unsupported camera matrix");
  const auto n=g.distortion.size();
  if(n!=4&&n!=5&&n!=8&&n!=12&&n!=14)throw std::runtime_error("invalid distortion count");
  const double sx=double(size.width)/g.calibration_width,sy=double(size.height)/g.calibration_height;
  if(size!=cv::Size(g.calibration_width,g.calibration_height)&&
     (!g.allow_uniform_resize||std::abs(sx-sy)>1e-9))
    throw std::runtime_error("calibration/video resolution mismatch; only explicit uniform resizing is supported");
  camera_=cv::Mat(3,3,CV_64F,const_cast<double*>(g.camera_matrix.data())).clone();
  camera_.row(0)*=sx;camera_.row(1)*=sy;
  const cv::Mat distortion(g.distortion,true);
  cv::initUndistortRectifyMap(camera_,distortion,cv::Mat(),camera_,size,CV_32FC1,map_x_,map_y_);
  if(!cv::checkRange(map_x_)||!cv::checkRange(map_y_))
    throw std::runtime_error("calibration produced nonfinite remap coordinates");
}
cv::Mat ImagePreprocessor::process(const cv::Mat& image)const{
  if(image.empty()||image.size()!=size_)throw std::runtime_error("preprocessing input size changed");
  if(!rectified())return image;
  cv::Mat result;
  cv::remap(image,result,map_x_,map_y_,cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar());
  return result;
}
}
