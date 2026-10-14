#pragma once
#include <filesystem>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <string>
namespace outpost {
class RunOutput {
 public:
  explicit RunOutput(const std::filesystem::path& requested={},const std::filesystem::path& base="outputs"){
    namespace fs=std::filesystem;
    if(!requested.empty()){
      if(!requested.parent_path().empty())fs::create_directories(requested.parent_path());
      if(!fs::create_directory(requested))throw std::runtime_error("output directory already exists; refusing overwrite");
      root_=fs::absolute(requested);
    }else{
      fs::create_directories(base);
      const auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      for(int i=0;i<1000;++i){auto candidate=base/("run-"+std::to_string(stamp)+"-"+std::to_string(i));
        if(fs::create_directory(candidate)){root_=fs::absolute(candidate);break;}}
      if(root_.empty())throw std::runtime_error("cannot reserve unique run directory");
    }
    status_.open(path("run_status.txt"));
    if(!status_)throw std::runtime_error("cannot create run status");
    status_<<"state=in_progress\n";status_.flush();
    if(!status_)throw std::runtime_error("cannot write run status");
  }
  std::string path(const std::string& file)const{
    const std::filesystem::path leaf(file);
    if(file.empty()||leaf.is_absolute()||leaf.has_parent_path()||file=="."||file=="..")
      throw std::runtime_error("output name must be a filename");
    return (root_/leaf).string();
  }
  const std::filesystem::path& root()const{return root_;}
  void finish(long frames,bool interrupted,bool video){
    status_<<"frames="<<frames<<"\nvideo_enabled="<<video<<"\nstate="
           <<(frames==0?"empty_input":interrupted?"interrupted":"completed")<<'\n';
    status_.flush();if(!status_)throw std::runtime_error("cannot finalize run status");
  }
 private:
  std::filesystem::path root_;std::ofstream status_;
};
}
