#include "core.hpp"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try{
 char name[]="/tmp/outpost-video-input-XXXXXX";auto dir=mkdtemp(name);require(dir,"fixture directory unavailable");
 const std::filesystem::path root(dir);
 struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove_all(path,e);}} cleanup{root};
 auto complete=root/"complete.avi",truncated=root/"truncated.avi";
 cv::VideoWriter writer(complete.string(),cv::VideoWriter::fourcc('M','J','P','G'),30,{160,120});
 require(writer.isOpened(),"fixture writer unavailable");cv::RNG random(42);
 for(int i=0;i<40;++i){cv::Mat image(120,160,CV_8UC3);random.fill(image,cv::RNG::UNIFORM,0,256);writer.write(image);}writer.release();
 outpost::VideoSource valid(complete.string());require(valid.reported_frame_count()==40,"fixture metadata missing");
 outpost::Frame f;long count=0;while(valid.next(f)){require(f.index==count&&std::abs(f.time_seconds-count/30.)<1e-10,"timestamp/index changed");++count;}
 require(count==40&&!valid.next(f)&&f.image.empty(),"valid EOF not stable");
 const auto timestamps=root/"timestamps.csv";
 {std::ofstream out(timestamps);out.precision(17);out<<"frame,time_seconds\n";
  for(int i=0;i<40;++i)out<<i<<','<<i/30.+.001*i*i<<'\n';}
 for(int capacity:{0,1,2,8}){
   outpost::VideoSource ordered(complete.string(),timestamps.string(),capacity),reference(complete.string());
   outpost::Frame expected;int index=0;
   while(ordered.next(f)){require(reference.next(expected),"prefetch added frames");
     require(f.index==index&&cv::norm(f.image,expected.image,cv::NORM_INF)==0,"prefetch reordered/changed pixels");
     require(std::abs(f.time_seconds-(index/30.+.001*index*index))<1e-12,"sidecar time replaced by fps or processing time");++index;}
   require(index==40&&ordered.queue_high_water()<=size_t(capacity),"prefetch lost frames or exceeded bound");
   require(ordered.forecast_frame(0,3/30.+.009)==3,"exact VFR future timestamp not matched");
   require(ordered.forecast_frame(0,.1)==-1,"nearest VFR frame incorrectly substituted");
   require(ordered.forecast_frame(39,.1)==-2,"future beyond timestamp EOF not reported");
   require(std::abs(ordered.frame_interval(2)-(1/30.+.005))<1e-12,"VFR display interval wrong");
   require(!ordered.next(f)&&f.image.empty(),"prefetch EOF unstable");
 }
 // Cancellation must join a producer blocked by a full bounded queue.
 {outpost::VideoSource cancelled(complete.string(),"",1);require(cancelled.next(f),"prefetch failed first frame");}
 for(int bad=0;bad<4;++bad){
   {std::ofstream out(timestamps);out<<"frame,time_seconds\n";
    for(int i=0;i<(bad==0?39:40);++i)out<<i<<','<<(bad==1&&i==5?0:bad==2&&i==0?-1:i/30.)<<(bad==3?",junk":"")<<'\n';}
   bool failed=false;try{outpost::VideoSource rejected(complete.string(),timestamps.string(),2);}catch(...){failed=true;}
   require(failed,"invalid timestamp input accepted");
 }
 std::filesystem::copy_file(complete,truncated);
 std::filesystem::resize_file(truncated,std::filesystem::file_size(truncated)/2);
 outpost::VideoSource broken(truncated.string());require(broken.reported_frame_count()==40,"truncated fixture lost header count");
 bool rejected=false;try{while(broken.next(f)){};}catch(const std::exception& e){rejected=std::string(e.what()).find("video decode count mismatch")!=std::string::npos;}
 require(rejected,"truncated video reported clean EOF");
 {outpost::VideoSource async_broken(truncated.string(),"",2);bool failed=false;
  try{while(async_broken.next(f)){};}catch(...){failed=true;}
  require(failed,"prefetch swallowed decoder exception");}
 std::cout<<"valid EOF, repeated EOF and truncated-input contracts passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
