#include <atomic>
#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "sensor_sync.hpp"
#include "thread_safe_queue.hpp"
namespace fs = std::filesystem;
struct Config { std::size_t queue_capacity; long long max_delta_us; fs::path camera_file, imu_file, output_file; };
std::string trim(const std::string& s) {
  std::size_t b=0,e=s.size();
  while(b<e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while(e>b && std::isspace(static_cast<unsigned char>(s[e-1]))) --e;
  return s.substr(b,e-b);
}
long long nonnegative(const std::string& value,const std::string& name) {
  if(value.empty() || value.front()=='-') throw std::runtime_error(name+" must be a nonnegative integer");
  std::size_t used=0; long long n=0;
  try { n=std::stoll(value,&used); } catch(...) { throw std::runtime_error(name+" must be a nonnegative integer"); }
  if(used!=value.size() || n<0) throw std::runtime_error(name+" must be a nonnegative integer");
  return n;
}
Config read_config(const fs::path& path) {
  std::ifstream in(path); if(!in) throw std::runtime_error("cannot open config file: "+path.string());
  std::map<std::string,std::string> v; std::string line; std::size_t line_no=0;
  while(std::getline(in,line)) {
    ++line_no; line=trim(line); if(line.empty() || line.front()=='#') continue;
    auto pos=line.find('='); if(pos==std::string::npos) throw std::runtime_error("invalid config line "+std::to_string(line_no));
    auto key=trim(line.substr(0,pos)); auto value=trim(line.substr(pos+1));
    if(key.empty() || value.empty()) throw std::runtime_error("invalid config line "+std::to_string(line_no));
    if(key!="queue_capacity" && key!="max_delta_us" && key!="camera_file" && key!="imu_file" && key!="output_file")
      throw std::runtime_error("unknown config key: "+key);
    if(!v.emplace(key,value).second) throw std::runtime_error("duplicate config key: "+key);
  }
  for(const char* key:{"queue_capacity","max_delta_us","camera_file","imu_file","output_file"})
    if(!v.count(key)) throw std::runtime_error(std::string("missing config key: ")+key);
  long long capacity=nonnegative(v.at("queue_capacity"),"queue_capacity");
  if(capacity==0) throw std::runtime_error("queue_capacity must be greater than zero");
  fs::path base=fs::absolute(path).parent_path();
  return {static_cast<std::size_t>(capacity),nonnegative(v.at("max_delta_us"),"max_delta_us"),
          base/v.at("camera_file"),base/v.at("imu_file"),base/v.at("output_file")};
}
void produce_file(const fs::path& path,SensorType type,ThreadSafeQueue<SensorData>& queue) {
  std::ifstream in(path); if(!in) throw std::runtime_error("cannot open data file: "+path.string());
  std::string line; std::size_t line_no=0; long long previous=0; bool first=true;
  while(std::getline(in,line)) {
    ++line_no; std::istringstream parser(line); long long seq=-1,time=-1; std::string extra;
    if(line.empty() || !(parser>>seq>>time) || (parser>>extra) || seq<0 ||
       seq>std::numeric_limits<std::uint32_t>::max() || time<0)
      throw std::runtime_error(path.string()+":"+std::to_string(line_no)+": invalid packet");
    if(!first && time<previous) throw std::runtime_error(path.string()+":"+std::to_string(line_no)+": timestamps are not nondecreasing");
    first=false; previous=time;
    if(!queue.push({type,static_cast<std::uint32_t>(seq),time})) return;
  }
}
void write_results(const fs::path& path,const std::vector<SyncResult>& results) {
  std::ofstream out(path); if(!out) throw std::runtime_error("cannot open output file: "+path.string());
  for(const auto& r:results) {
    out<<"CAMERA "<<r.camera.sequence;
    if(r.imu) out<<" IMU "<<r.imu->sequence<<" DELTA "<<r.delta_us; else out<<" UNMATCHED";
    out<<'\n';
  }
}
int main(int argc,char* argv[]) {
  if(argc!=2) { std::cerr<<"Usage: "<<argv[0]<<" <config-file>\n"; return 1; }
  try {
    Config cfg=read_config(argv[1]); ThreadSafeQueue<SensorData> queue(cfg.queue_capacity);
    std::vector<SensorData> cameras,imus; std::atomic<int> remaining{2};
    std::mutex error_mutex; std::exception_ptr thread_error;
    auto report_error=[&]{ std::lock_guard<std::mutex> lock(error_mutex); if(!thread_error) thread_error=std::current_exception(); queue.close(); };
    auto producer=[&](const fs::path& path,SensorType type){
      try { produce_file(path,type,queue); } catch(...) { report_error(); }
      if(remaining.fetch_sub(1)==1) queue.close();
    };
    std::thread consumer([&]{
      try { SensorData data{}; while(queue.pop(data)) (data.type==SensorType::CAMERA?cameras:imus).push_back(data); }
      catch(...) { report_error(); }
    });
    std::thread camera_thread(producer,cfg.camera_file,SensorType::CAMERA);
    std::thread imu_thread(producer,cfg.imu_file,SensorType::IMU);
    camera_thread.join(); imu_thread.join(); consumer.join();
    if(thread_error) std::rethrow_exception(thread_error);
    auto results=synchronize(cameras,imus,cfg.max_delta_us); write_results(cfg.output_file,results);
    std::cout<<"Processed "<<cameras.size()<<" camera packets and "<<imus.size()<<" IMU packets.\n";
    std::cout<<"Wrote "<<results.size()<<" results to "<<cfg.output_file<<'\n';
    return 0;
  } catch(const std::exception& e) { std::cerr<<"Error: "<<e.what()<<'\n'; return 1; }
}
