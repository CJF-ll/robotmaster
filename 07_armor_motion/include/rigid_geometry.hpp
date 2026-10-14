#pragma once
#include <opencv2/calib3d.hpp>
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>
namespace outpost {
// Pure geometry for synthetic contracts / a future calibrated estimator.
// State must be supplied by an estimator, never invented from demo constants.
// Target coordinates: Z is spin axis; X/Y radial plane. Angles in radians, lengths in m.
struct RigidSlot {double phase,height,radius;};
struct RigidState {cv::Vec3d center,velocity;double phase=0,omega=0;};
class RigidTargetGeometry {
 public:
  RigidTargetGeometry(double width,double height,double pitch,std::vector<RigidSlot> slots)
    :width_(width),height_(height),pitch_(pitch),slots_(std::move(slots)){
    if(!std::isfinite(width)||width<=0||!std::isfinite(height)||height<=0||
       !std::isfinite(pitch)||slots_.empty())throw std::runtime_error("invalid rigid dimensions");
    for(const auto& s:slots_)if(!std::isfinite(s.phase)||!std::isfinite(s.height)||
       !std::isfinite(s.radius)||s.radius<=0)throw std::runtime_error("invalid rigid slot");
  }
  std::array<cv::Point3d,4> corners(const RigidState& s,size_t slot,double dt)const{
    if(slot>=slots_.size()||!std::isfinite(dt)||dt<0||!std::isfinite(s.phase)||!std::isfinite(s.omega))
      throw std::runtime_error("invalid rigid state/time/slot");
    for(int i=0;i<3;++i)if(!std::isfinite(s.center[i])||!std::isfinite(s.velocity[i]))
      throw std::runtime_error("nonfinite rigid state");
    const auto& q=slots_[slot];const double a=std::remainder(s.phase+s.omega*dt+q.phase,2*CV_PI);
    const cv::Vec3d radial(std::cos(a),std::sin(a),0),tangent(-std::sin(a),std::cos(a),0);
    const cv::Vec3d up=radial*std::sin(pitch_)+cv::Vec3d(0,0,std::cos(pitch_));
    const cv::Vec3d c=s.center+s.velocity*dt+q.radius*radial+cv::Vec3d(0,0,q.height);
    std::array<cv::Point3d,4> out;const int signs[4][2]={{-1,1},{1,1},{1,-1},{-1,-1}};
    for(int i=0;i<4;++i){const auto p=c+signs[i][0]*width_*.5*tangent+signs[i][1]*height_*.5*up;
      out[i]={p[0],p[1],p[2]};}return out;
  }
 private:double width_,height_,pitch_;std::vector<RigidSlot> slots_;
};
}
