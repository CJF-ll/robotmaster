#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Error {double distance,total,along,cross;bool moving;};
Error decompose(double ox,double oy,double px,double py,double tx,double ty,double gate){
  const double dx=tx-ox,dy=ty-oy,ex=px-tx,ey=py-ty;
  const double d=std::hypot(dx,dy),e=std::hypot(ex,ey);
  if(d<=gate)return {d,e,0,0,false};
  return {d,e,(ex*dx+ey*dy)/d,(dx*ey-dy*ex)/d,true};
}
std::vector<std::string> split(const std::string& line){
  std::vector<std::string> fields;size_t start=0;
  for(;;){const auto end=line.find(',',start);fields.push_back(line.substr(start,end-start));
    if(end==std::string::npos)return fields;
    start=end+1;}
}
double number(const std::string& text){size_t used=0;double v=std::stod(text,&used);
  if(used!=text.size()||!std::isfinite(v))throw std::runtime_error("invalid finite number");
  return v;}
long integer(const std::string& text){size_t used=0;long v=std::stol(text,&used);
  if(used!=text.size()||v<0)throw std::runtime_error("invalid nonnegative integer");
  return v;}
struct Row {long frame,target,id;std::string status;Error error{};};
Row parse(const std::string& text,double gate){
  const auto f=split(text);if(f.size()!=13)throw std::runtime_error("expected 13 audit columns");
  Row r{integer(f[0]),integer(f[1]),integer(f[2]),f[3],{}};
  if(r.target<=r.frame)throw std::runtime_error("nonfuture target");
  static const std::set<std::string> states={"scored","no_accepted_target","interrupted_history","timestamp_skipped","end_of_video"};
  if(!states.count(r.status))throw std::runtime_error("unknown audit status");
  const double ox=number(f[4]),oy=number(f[5]),px=number(f[6]),py=number(f[7]);
  if(r.status=="scored"){
    r.error=decompose(ox,oy,px,py,number(f[8]),number(f[9]),gate);
    if(std::abs(number(f[10])-r.error.total)>1e-4||std::abs(number(f[11])-r.error.distance)>1e-4)
      throw std::runtime_error("audit errors disagree with coordinates");
    if(!f[12].empty())number(f[12]);
  }else for(size_t i=8;i<13;++i)if(!f[i].empty())throw std::runtime_error("unscored row contains scores");
  return r;
}
struct Group {long count=0;double sum=0,maximum=0;
  void add(double e){++count;sum+=e;maximum=std::max(maximum,e);}};
struct Episode {long id,first,last,count=0;double sum=0,maximum=0,along=0,cross=0;};
void check(bool value){if(!value)throw std::runtime_error("self-test failed");}
void self_test(){
  auto a=decompose(0,0,7,0,10,0,2);check(a.moving&&a.along==-3&&a.cross==0);
  a=decompose(0,0,13,4,10,0,2);check(a.along==3&&a.cross==4&&a.total==5);
  a=decompose(0,0,0,7,0,10,2);check(a.along==-3&&a.cross==0);
  a=decompose(0,0,-13,0,-10,0,2);check(a.along==3);
  a=decompose(0,0,50,0,0,0,2);check(!a.moving&&a.total==50);
  check(!decompose(0,0,3,0,2,0,2).moving);
  check(parse("0,3,1,scored,0,0,7,0,10,0,3,10,",2).error.along==-3);
  check(parse("0,3,1,no_accepted_target,0,0,7,0,,,,,",2).status=="no_accepted_target");
  for(const auto& bad:{"0,3,1,scored,0,0,7,0,10,0,2,10,","0,3,1,unknown,0,0,7,0,,,,,",
                      "0,3,1,scored,0,0,nan,0,10,0,3,10,","3,0,1,no_accepted_target,0,0,7,0,,,,,"}){
    bool rejected=false;try{parse(bad,2);}catch(const std::exception&){rejected=true;}check(rejected);
  }
  std::cout<<"direction decomposition and audit validation passed\n";
}
int main(int argc,char** argv){try{
  if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
  if(argc!=4){std::cerr<<"usage: prediction_diagnostics audit.csv moving_distance_px high_error_px\n";return 2;}
  const double gate=number(argv[2]),high=number(argv[3]);
  if(gate<=0||high<=0)throw std::runtime_error("thresholds must be positive");
  std::ifstream file(argv[1]);if(!file)throw std::runtime_error("cannot read audit");
  std::string line;std::getline(file,line);
  if(line!="frame,target_frame,id,status,origin_x,origin_y,predicted_x,predicted_y,observed_x,observed_y,error_px,hold_error_px,shadow_error_px")
    throw std::runtime_error("unexpected audit schema");
  std::map<std::string,Group> groups;std::map<std::string,long> statuses;
  std::map<long,size_t> active;std::vector<Episode> episodes;
  std::set<std::pair<long,long>> keys;long last=-1,lead=-1;
  while(std::getline(file,line)){
    const auto r=parse(line,gate);
    if(r.frame<last||!keys.emplace(r.frame,r.id).second)throw std::runtime_error("unordered/duplicate audit row");
    if(lead<0)lead=r.target-r.frame;
    if(r.target-r.frame!=lead)throw std::runtime_error("mixed forecast horizons");
    last=r.frame;++statuses[r.status];
    if(r.status!="scored"){active.erase(r.id);continue;}
    const auto& e=r.error;groups["all_scored"].add(e.total);
    std::string category="low_displacement_direction_undefined";
    if(e.moving)category=std::abs(e.cross)>std::abs(e.along)?"lateral_dominant":
                           e.along<0?"along_shortfall":"along_overshoot";
    groups[category].add(e.total);
    if(e.total<high){active.erase(r.id);continue;}
    groups["high_error/"+category].add(e.total);
    auto found=active.find(r.id);
    if(found==active.end()||episodes[found->second].last+1!=r.frame){
      active[r.id]=episodes.size();episodes.push_back({r.id,r.frame,r.frame});
    }
    auto& p=episodes[active[r.id]];p.last=r.frame;++p.count;p.sum+=e.total;
    p.maximum=std::max(p.maximum,e.total);p.along+=e.along;p.cross+=std::abs(e.cross);
  }
  if(file.bad()||keys.empty())throw std::runtime_error("empty or unreadable audit");
  std::cout<<std::fixed<<std::setprecision(3)<<"horizon_frames="<<lead<<" moving_gate_px="<<gate<<" high_error_px="<<high<<'\n';
  for(const auto& [name,n]:statuses)std::cout<<"status "<<name<<" n="<<n<<'\n';
  for(const auto& [name,g]:groups)std::cout<<name<<" n="<<g.count<<" mean="<<g.sum/g.count<<" max="<<g.maximum<<'\n';
  std::sort(episodes.begin(),episodes.end(),[](const auto& a,const auto& b){return a.sum>b.sum;});
  for(const auto& p:episodes)std::cout<<"episode id="<<p.id<<" issued_frames="<<p.first<<".."<<p.last
    <<" n="<<p.count<<" mean="<<p.sum/p.count<<" max="<<p.maximum<<'\n';
  std::cout<<"Along signs describe errors relative to later detector displacement, not proven causes.\n"
              "Low-displacement direction is undefined. Unscored rows are not successes.\n"
              "Retrospective diagnostics only; no future measurements feed the tracker.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
