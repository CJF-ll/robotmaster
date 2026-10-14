#include "core.hpp"
#include "detection_backend.hpp"
#include "target_model.hpp"
#include "rigid_geometry.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>
using namespace outpost;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv){
  if(argc!=2)return 2;auto c=Config::load(argv[1]);
  auto backend=make_detector("traditional",c.detector);Detector legacy(c.detector);
  auto model=make_target_model("image_2d",c.tracker,c.slots);Tracker tracker(c.tracker);TargetSlots slots(c.slots);
  for(int f=0;f<40;++f){cv::Mat im(160,240,CV_8UC3,cv::Scalar());
    if(f!=20)for(int x:{40+f,70+f})cv::rectangle(im,{x,50,4,20},{0,80,255},cv::FILLED);
    const auto batch=backend->detect(im);const auto old=legacy.detect(im);
    check(batch.observations.size()==old.size()&&batch.lamps.size()==legacy.bars().size(),"backend changed detection");
    const auto a=model->update(f/30.,batch.observations,.1);const auto b=tracker.step(f/30.,old,.1);
    const auto ids=slots.update(f/30.,b,old);
    check(!a.physical_state_available&&a.tracks.size()==b.size(),"adapter claims physical state/changes tracks");
    for(size_t i=0;i<b.size();++i)check(a.tracks[i].id==b[i].id&&a.tracks[i].future_center==b[i].future_center&&
      a.tracks[i].future_valid==b[i].future_valid&&a.slots[i].slot==ids[i].slot,"adapter changed state");
  }
  bool refused=false;try{make_detector("unavailable",c.detector);}catch(...){refused=true;}
  check(refused,"missing backend silently replaced");
  refused=false;try{make_target_model("calibrated_3d",c.tracker,c.slots);}catch(...){refused=true;}
  check(refused,"unimplemented estimator silently replaced");
  const std::vector<RigidSlot> spec{{0,0,.25},{2*CV_PI/3,.03,.25},{4*CV_PI/3,.07,.25}};
  RigidTargetGeometry g(.135,.056,0,spec);RigidState state{{0,0,2},{.1,0,0},0,.7};
  auto center=[](const auto& points){cv::Point3d p;for(const auto& q:points)p+=q;return p*.25;};
  for(size_t id=0;id<3;++id){const auto p=g.corners(state,id,0),q=g.corners(state,id,.2);
    check(std::abs(cv::norm(p[1]-p[0])-.135)<1e-10&&std::abs(cv::norm(p[3]-p[0])-.056)<1e-10,"rigid dimensions changed");
    check(std::abs(center(p).z-(2+spec[id].height))<1e-10,"staggered height lost");
    check(std::abs(center(q).x-(.02+.25*std::cos(.14+spec[id].phase)))<1e-10,"phase prediction wrong");
  }
  const auto original=state;g.corners(state,0,1);check(state.phase==original.phase&&state.center==original.center,"forecast mutated state");
  auto wrapped=state;wrapped.phase+=2*CV_PI;
  check(cv::norm(g.corners(state,0,0)[0]-g.corners(wrapped,0,0)[0])<1e-10,"phase wrapping broken");
  refused=false;try{RigidTargetGeometry bad(.1,.1,0,{{0,0,-1}});}catch(...){refused=true;}
  check(refused,"invalid radius accepted");
}
