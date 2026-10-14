#include "estimator.hpp"
namespace outpost {
namespace {
class ImageEstimator final:public Estimator {
 public:
  explicit ImageEstimator(const Config& c):model_(c.tracker,c.slots){}
  EstimatorOutput update(double t,const std::vector<Detection>& d,double h)override{
    EstimatorOutput result;result.image=model_.update(t,d,h);result.epoch=result.image.epoch;return result;
  }
  void reconfigure(const Config& c,cv::Size)override{
    require_estimator_mode(c,mode(),"image estimator reconfiguration");model_.reconfigure(c.tracker,c.slots);
  }
  const char* mode()const override{return "image_2d";}
 private: ImageTargetModel model_;
};
class RigidEstimator final:public Estimator {
 public:
  RigidEstimator(const Config& c,cv::Size size):model_(calibrated_config(c,size)){}
  EstimatorOutput update(double t,const std::vector<Detection>& d,double h)override{
    EstimatorOutput result;result.rigid=model_.update(t,d,h);
    if(result.rigid->initialized&&result.rigid->generation!=generation_){
      if(generation_)++epoch_;
      generation_=result.rigid->generation;
    }
    result.epoch=epoch_;return result;
  }
  void reconfigure(const Config& c,cv::Size size)override{
    require_estimator_mode(c,mode(),"rigid estimator reconfiguration");
    CalibratedRigidEstimator prepared(calibrated_config(c,size));
    model_=std::move(prepared);generation_=0;++epoch_;
  }
  const char* mode()const override{return "calibrated_3d";}
 private: CalibratedRigidEstimator model_;int epoch_=0;std::uint64_t generation_=0;
};
}
void require_estimator_mode(const Config& c,const std::string& expected,const std::string& tool){
  if(c.geometry.mode!=expected)throw std::runtime_error(tool+" supports "+expected+
    " only; requested "+c.geometry.mode+"; refusing to evaluate a different estimator");
}
std::unique_ptr<Estimator> make_estimator(const Config& c,cv::Size size){
  if(c.geometry.mode=="image_2d")return std::make_unique<ImageEstimator>(c);
  if(c.geometry.mode=="calibrated_3d")return std::make_unique<RigidEstimator>(c,size);
  throw std::runtime_error("unknown estimator mode: "+c.geometry.mode);
}
}
