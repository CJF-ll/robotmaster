#include <yaml-cpp/yaml.h>
#include "forecast_assessment.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <map>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cerrno>
#include <cmath>
namespace fs=std::filesystem;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
std::string read(const fs::path& path){std::ifstream f(path);return std::string(std::istreambuf_iterator<char>(f),{});}
template<class F>void await(F condition){
 const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
 while(!condition()){require(std::chrono::steady_clock::now()<deadline,"main-process handshake timed out");std::this_thread::sleep_for(std::chrono::milliseconds(2));}
}
std::vector<std::string> fields(const std::string& row){std::stringstream s(row);std::string f;std::vector<std::string> v;while(std::getline(s,f,','))v.push_back(f);return v;}
int main(int argc,char**argv){try{
 require(argc==3||argc==4,"usage: live_reload_tests base.yaml outpost_video [calibrated|prefetch]");
 const std::string mode=argc==4?argv[3]:"image";
 const bool calibrated=mode=="calibrated",prefetch=mode=="prefetch";
 require(mode=="image"||calibrated||prefetch,"unknown test mode");
 signal(SIGPIPE,SIG_IGN);
 char name[]="/tmp/outpost-live-reload-XXXXXX";auto made=mkdtemp(name);require(made,"cannot create fixture directory");
 struct Cleanup {fs::path path;pid_t child=-1;~Cleanup(){if(child>0){kill(child,SIGKILL);waitpid(child,nullptr,0);}if(std::uncaught_exceptions()){std::cerr<<"fixture retained: "<<path<<'\n';return;}std::error_code e;fs::remove_all(path,e);}} cleanup{made};
 const auto root=cleanup.path,video=root/"synthetic.avi",fifo=root/"config.pipe",output=root/"run";
 const cv::Size image_size=calibrated?cv::Size(1280,720):cv::Size(160,120);
 cv::VideoWriter writer(video.string(),cv::VideoWriter::fourcc('F','F','V','1'),30,image_size);require(writer.isOpened(),"fixture writer unavailable");
 cv::Mat pixels(image_size,CV_8UC3,cv::Scalar());for(int x:{30,60})cv::rectangle(pixels,{x,40,4,20},{0,80,255},cv::FILLED);
 if(calibrated){
   pixels.setTo(cv::Scalar());const double phase=.5;
   const cv::Mat k=(cv::Mat_<double>(3,3)<<1100,0,640,0,1100,360,0,0,1);
   const cv::Matx33d basis(0,1,0,0,0,-1,-1,0,0);
   const cv::Vec3d radial(std::cos(phase),std::sin(phase),0),tangent(-std::sin(phase),std::cos(phase),0),up(0,0,1);
   for(int side:{-1,1}){
     std::vector<cv::Point3d> points;
     for(const auto& sign:std::vector<cv::Point2d>{{-1,1},{1,1},{1,-1},{-1,-1}}){
       const auto p=basis*(cv::Vec3d(-2.5,0,.2)+.27*radial+(side*.0675+sign.x*.004)*tangent+sign.y*.03*up);
       points.emplace_back(p[0],p[1],p[2]);
     }
     std::vector<cv::Point2d> projected;cv::projectPoints(points,cv::Vec3d(),cv::Vec3d(),k,cv::Mat::zeros(1,5,CV_64F),projected);
     std::vector<cv::Point> q;for(auto p:projected)q.emplace_back(cvRound(p.x),cvRound(p.y));cv::fillConvexPoly(pixels,q,{0,80,255});
   }
 }
 for(int i=0;i<12;++i)writer.write(pixels);writer.release();
 auto base=YAML::LoadFile(argv[1]);base["input"]["path"]=video.string();base["detector"]["backend"]="traditional";
 if(prefetch){const auto timestamps=root/"timestamps.csv";std::ofstream out(timestamps);out.precision(17);
   out<<"frame,time_seconds\n";for(int i=0;i<12;++i)out<<i<<','<<i/30.+.0005*i*i<<'\n';
   out.close();require(bool(out),"cannot write fixture timestamps");
   base["input"]["timestamp_file"]=timestamps.string();base["input"]["prefetch_frames"]=2;}
 base["geometry"]["undistort"]=false;base["prediction"]["horizon_seconds"]=.1;
 base["tracking"]["confirmation_hits"]=3;base["tracking"]["prediction_hits"]=5;
 if(calibrated){auto g=base["geometry"];
   g["mode"]="calibrated_3d";g["calibration_width"]=1280;g["calibration_height"]=720;
   g["camera_matrix"]=std::vector<double>{1100,0,640,0,1100,360,0,0,1};g["distortion"]=std::vector<double>{0,0,0,0,0};
   g["camera_from_target_rotation"]=std::vector<double>{0,1,0,0,0,-1,-1,0,0};
   g["lightbar_length_m"]=.06;g["lightbar_center_distance_m"]=.135;g["rotation_radius_m"]=.27;
   g["armor_pitch_deg"]=0;g["slot_height_offsets_m"]=std::vector<double>{0,0,0};
 }
 require(mkfifo(fifo.c_str(),0600)==0,"cannot create config FIFO");
 const auto executable=fs::absolute(argv[2]).string();cleanup.child=fork();require(cleanup.child>=0,"cannot launch main");
 if(cleanup.child==0){
   const int log=open((root/"process.log").c_str(),O_CREAT|O_WRONLY|O_TRUNC,0600);if(log>=0){dup2(log,1);dup2(log,2);close(log);}
   execl(executable.c_str(),executable.c_str(),fifo.c_str(),"--headless","--no-video","--output-dir",output.c_str(),static_cast<char*>(nullptr));_exit(127);
 }
 auto send=[&](const std::string& text){int fd=-1;
   await([&]{fd=open(fifo.c_str(),O_WRONLY|O_NONBLOCK);if(fd>=0)return true;require(errno==ENXIO||errno==EINTR,"cannot open fixture FIFO");return false;});
   size_t n=0;while(n<text.size()){const auto written=write(fd,text.data()+n,text.size()-n);if(written<=0){close(fd);throw std::runtime_error("cannot write fixture config");}n+=size_t(written);}close(fd);
 };
 // FIFO EOF completes one poll. Audit flush acknowledges that this frame has
 // consumed it before the next edit; no scheduler-dependent sleep/edit race.
 send(YAML::Dump(base));await([&]{return read(output/"config_events.csv").find(",startup,")!=std::string::npos;});
 for(int frame=0;frame<12;++frame){auto cfg=YAML::Clone(base);
   if(frame>=3)cfg["prediction"]["horizon_seconds"]=.15;
   if(frame==4||frame==5)cfg["prediction"]["horizon_seconds"]=-1;
   if(calibrated){
     if(frame>=3)cfg["tracking"]["measurement_noise"]=12; // Inactive in 3D: must preserve epoch.
     if(frame>=7)cfg["rigid_estimator"]["max_center_error_m"]=.12;
   }else if(frame>=7)cfg["tracking"]["acceleration_noise"]=30000;
   if(frame>=8)cfg["prediction"]["horizon_seconds"]=.2;
   if(frame==10)cfg["input"]["path"]="not-the-current-video.mp4";
   send(YAML::Dump(cfg)+"\n# integration frame "+std::to_string(frame)+"\n");
   await([&]{return read(output/"config_events.csv").find("\n"+std::to_string(frame)+",")!=std::string::npos;});
 }
 int status=0;await([&]{return waitpid(cleanup.child,&status,WNOHANG)==cleanup.child;});cleanup.child=-1;
 if(!WIFEXITED(status)||WEXITSTATUS(status)!=0)std::cerr<<read(root/"process.log");
 require(WIFEXITED(status)&&WEXITSTATUS(status)==0,"main program failed");
 std::map<int,std::vector<std::string>> events;std::stringstream audit(read(output/"config_events.csv"));std::string line;std::getline(audit,line);
 while(std::getline(audit,line)){auto f=fields(line);require(f.size()>=6,"incomplete audit row");require(events.emplace(std::stoi(f[0]),f).second,"duplicate frame event");}
 require(events.size()==13,"missing startup/frame audit");
 for(int i=0;i<12;++i){const std::string expected=(i==4||i==5||i==10)?"rejected":i==7?"reset":"live";require(events.at(i)[3]==expected,"wrong update decision");}
 require(events.at(4)[4]==events.at(3)[4]&&events.at(5)[4]==events.at(3)[4]&&events.at(10)[4]==events.at(9)[4],"rejection replaced active snapshot");
 require(events.at(4)[2]!=events.at(5)[2],"different invalid edits lost provenance");
 const auto active=YAML::LoadFile((output/events.at(11)[4]).string());require(active["prediction"]["horizon_seconds"].as<double>()==.2,"final live update not applied");
 require(active["input"]["path"].as<std::string>()==video.string(),"restart-only input change committed");
 std::map<int,int> ids;
 if(!calibrated){std::stringstream tracking(read(output/"tracking.csv"));std::getline(tracking,line);
 while(std::getline(tracking,line)){auto f=fields(line);require(ids.emplace(std::stoi(f[0]),std::stoi(f[2])).second,"unexpected duplicate track");}
 for(int i=2;i<=6;++i)require(ids.at(i)==1,"live/rejected config reset a valid identity");
 require(!ids.count(7)&&!ids.count(8),"model change bypassed reconfirmation");
 for(int i=9;i<=11;++i)require(ids.at(i)==2,"model reset failed to create new monotonic identity");
 }else{
   std::stringstream states(read(output/"rigid_state.csv"));std::getline(states,line);int rows=0,used=0;
   while(std::getline(states,line)){const auto f=fields(line);const auto frame=std::stoi(f[0]);
     require(std::stoi(f[2])==(frame>=7?1:0),"inactive/rejected configuration changed rigid epoch");
     used+=std::stoi(f[5]);++rows;
   }
   require(rows==12&&used>=8,"calibrated live reload lacked actual measurement updates");
 }
 require(read(output/"run_status.txt").find("state=completed")!=std::string::npos,"run did not complete");
  require(!fs::exists(output/"tracking-preview.mp4"),"no-video mode wrote a video");
 std::stringstream timings(read(output/"timing.csv"));std::getline(timings,line);int timed_frames=0;
 while(std::getline(timings,line)){const auto f=fields(line);require(f.size()==11,"bad timing column count");
   require(std::stoi(f[0])==timed_frames++,"missing/duplicate timing frame");double sum=0;
   for(int i=1;i<=10;++i){const double v=std::stod(f[i]);require(std::isfinite(v)&&v>=0,"invalid timing value");if(i<10)sum+=v;}
   require(std::abs(sum-std::stod(f[10]))<1e-6,"timing stages do not sum to total");
 }
 require(timed_frames==12,"incomplete timing coverage");
 if(prefetch){std::stringstream times(read(output/"frame_timestamps.csv"));std::getline(times,line);int n=0;
   while(std::getline(times,line)){const auto f=fields(line);require(std::stoi(f[0])==n,"prefetch main frame ordering changed");
     require(std::abs(std::stod(f[1])-(n/30.+.0005*n*n))<1e-12,"main discarded explicit timestamps");++n;}
   require(n==12,"prefetch main lost frames");
   require(read(output/"input_metadata.txt").find("time_basis=timestamp_sidecar_seconds")!=std::string::npos,"main timestamp basis wrong");
   const auto pipeline=read(output/"input_pipeline.txt");require(pipeline.find("capacity=2")!=std::string::npos&&
      pipeline.find("drop_policy=never")!=std::string::npos,"missing bounded pipeline metadata");}
 std::stringstream detector_times(read(output/"detector_timing.csv"));std::getline(detector_times,line);int detector_frames=0;
 while(std::getline(detector_times,line)){const auto f=fields(line);require(f.size()==9,"bad detector timing column count");
   require(std::stoi(f[0])==detector_frames++,"missing/duplicate detector timing frame");
   for(int i=1;i<=8;++i){const double v=std::stod(f[i]);require(std::isfinite(v)&&v>=0,"invalid detector timing value");
     if(i>=2)require(v==0,"traditional backend reported neural work");}
 }
 require(detector_frames==12,"incomplete detector timing coverage");
 std::stringstream motion(read(output/"motion_state.csv"));std::getline(motion,line);int motion_rows=0;
 while(std::getline(motion,line)){const auto f=fields(line);require(f.size()==22,"bad forecast diagnostic column count");
   const auto blocks=unsigned(std::stoul(f[18]));
   require((std::stoi(f[13])!=0)==(blocks==0),"forecast mask disagrees with validity");
   require(f[19]==outpost::forecast_block_names(blocks),"forecast text disagrees with mask");
   const double deviation=std::stod(f[17]);
   require(std::isfinite(deviation)&&std::abs(deviation-std::sqrt(std::max(std::stod(f[11]),std::stod(f[12]))))<1e-9,
           "forecast uncertainty not derived from logged covariance");
   const std::string reason=std::stoi(f[14])?"drawn":!std::stoi(f[3])?"no_accepted_observation":
                            !std::stoi(f[13])?"model_forecast_blocked":"visibility_gate";
   require(f[20]==reason,"model validity conflated with render permission");
   require(std::isfinite(std::stod(f[21]))&&std::stod(f[21])>=0,"invalid logged horizon");++motion_rows;
 }
 require(motion_rows==int(ids.size()),"forecast diagnostics missing tracked rows");
 std::cout<<"main-process reload: live, rejection, recovery, model reset and identity contracts passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
