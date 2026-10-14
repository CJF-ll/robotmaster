#include "video_runner.hpp"
#include <iostream>
#include <string>
int main(int argc,char** argv){
  if(argc<2){
    std::cerr<<"usage: outpost_video config.yaml [--headless] [--no-video] [--output-dir NEW_DIRECTORY]\n";return 2;
  }
  bool headless=false,save_video=true;std::string directory;
  for(int i=2;i<argc;++i){
    const std::string option=argv[i];
    if(option=="--headless"&&!headless)headless=true;
    else if(option=="--no-video"&&save_video)save_video=false;
    else if(option=="--output-dir"&&directory.empty()&&i+1<argc){
      directory=argv[++i];if(directory.empty())return 2;
    }else{std::cerr<<"invalid or duplicate option: "<<option<<'\n';return 2;}
  }
  return outpost::run_video(argv[1],headless,save_video,directory);
}
