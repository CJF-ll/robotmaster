#include "calibrated_video.hpp"
#include "video_runner.hpp"
#include "estimator.hpp"
namespace outpost {
int run_calibrated_video(const std::string& path,bool headless,bool save_video,const std::string& directory){
  require_estimator_mode(Config::load(path),"calibrated_3d","calibrated video");
  return run_video(path,headless,save_video,directory);
}
}
