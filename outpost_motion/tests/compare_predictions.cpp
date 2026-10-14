#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Row{long frame;double error,displacement;long target=0;double ox=0,oy=0,px=0,py=0,tx=0,ty=0;};
struct Run{int issued=0;std::map<std::string,int> statuses;std::map<std::string,Row> scored;};
std::vector<std::string> fields(const std::string& line){
  std::vector<std::string> out;std::stringstream s(line);std::string v;
  while(std::getline(s,v,','))out.push_back(v);
  if(!line.empty()&&line.back()==',')out.emplace_back();return out;
}
Run read(std::istream& input){
  Run run;std::string line;
  if(!std::getline(input,line)||line!="frame,target_frame,id,status,origin_x,origin_y,predicted_x,predicted_y,observed_x,observed_y,error_px,hold_error_px,shadow_error_px")
    throw std::runtime_error("incompatible prediction audit header");
  while(std::getline(input,line)){
    auto f=fields(line);if(f.size()!=13)throw std::runtime_error("malformed audit row");
    if(f[3]!="scored"&&f[3]!="no_accepted_target"&&f[3]!="interrupted_history"&&
       f[3]!="end_of_video"&&f[3]!="timestamp_skipped")throw std::runtime_error("unknown audit status");
    ++run.issued;++run.statuses[f[3]];if(f[3]!="scored")continue;
    // Physical IDs are unverified and local IDs can change between runs. Match
    // the SAME origin and target detector observations, not merely the frame/ID.
    const auto key=f[0]+","+f[1]+","+f[4]+","+f[5]+","+f[8]+","+f[9];
    const double error=std::stod(f[10]),displacement=std::stod(f[11]);
    if(!std::isfinite(error)||error<0||!std::isfinite(displacement)||displacement<0)
      throw std::runtime_error("invalid audit error");
    Row row{std::stol(f[0]),error,displacement};row.target=std::stol(f[1]);
    row.ox=std::stod(f[4]);row.oy=std::stod(f[5]);row.px=std::stod(f[6]);row.py=std::stod(f[7]);
    row.tx=std::stod(f[8]);row.ty=std::stod(f[9]);
    for(double value:{row.ox,row.oy,row.px,row.py,row.tx,row.ty})
      if(!std::isfinite(value))throw std::runtime_error("nonfinite audit coordinate");
    if(!run.scored.emplace(key,row).second)
      throw std::runtime_error("duplicate scored observation pair");
  }
  return run;
}
void summary(const std::string& name,std::vector<double> errors){
  std::cout<<name<<" n="<<errors.size();if(errors.empty()){std::cout<<'\n';return;}
  std::sort(errors.begin(),errors.end());double sum=0,squares=0;
  for(auto e:errors){sum+=e;squares+=e*e;}
  std::cout<<std::fixed<<std::setprecision(3)<<" mean="<<sum/errors.size()<<" p95="
           <<errors[size_t(std::ceil(.95*errors.size()))-1]<<" rmse="<<std::sqrt(squares/errors.size())
           <<" max="<<errors.back()<<'\n';
}
int compare(const Run& baseline,const Run& candidate){
  std::map<std::string,std::pair<std::vector<double>,std::vector<double>>> groups;
  int shared=0,improved=0;
  for(const auto& [key,a]:baseline.scored){auto found=candidate.scored.find(key);if(found==candidate.scored.end())continue;
    const auto& b=found->second;++shared;if(b.error<a.error)++improved;
    for(const auto& group:{std::string("all"),std::string(a.frame<450?"before_450":"from_450"),
                          std::string(a.displacement>2?"displacement_gt_2px":"displacement_le_2px")}){
      groups[group].first.push_back(a.error);groups[group].second.push_back(b.error);
    }
  }
  std::cout<<"baseline_issued="<<baseline.issued<<" candidate_issued="<<candidate.issued
           <<" shared="<<shared<<" baseline_only="<<baseline.scored.size()-shared
           <<" candidate_only="<<candidate.scored.size()-shared<<" improved="<<improved<<'\n';
  for(const auto& [s,n]:baseline.statuses)std::cout<<"baseline "<<s<<'='<<n<<'\n';
  for(const auto& [s,n]:candidate.statuses)std::cout<<"candidate "<<s<<'='<<n<<'\n';
  for(const auto& [name,pair]:groups){summary(name+" baseline",pair.first);summary(name+" candidate",pair.second);}
  return shared;
}
using RowPair=std::pair<const Row*,const Row*>;
std::vector<RowPair> nearby_pairs(const Run& a,const Run& b,double tolerance){
  if(!std::isfinite(tolerance)||tolerance<=0)throw std::runtime_error("positive finite tolerance required");
  std::vector<RowPair> eligible;std::map<const Row*,int> degree_a,degree_b;
  for(const auto& [ka,x]:a.scored)for(const auto& [kb,y]:b.scored){
    if(x.frame!=y.frame||x.target!=y.target)continue;
    if(std::hypot(x.ox-y.ox,x.oy-y.oy)>tolerance||std::hypot(x.tx-y.tx,x.ty-y.ty)>tolerance)continue;
    eligible.emplace_back(&x,&y);++degree_a[&x];++degree_b[&y];
  }
  std::vector<RowPair> result;
  for(auto p:eligible)if(degree_a[p.first]==1&&degree_b[p.second]==1)result.push_back(p);
  return result;
}
int compare_nearby(const Run& a,const Run& b,double tolerance){
  const auto pairs=nearby_pairs(a,b,tolerance);
  std::vector<double> aa,ba,ab,bb;
  for(auto [x,y]:pairs){
    aa.push_back(std::hypot(x->px-x->tx,x->py-x->ty));
    ba.push_back(std::hypot(y->px-x->tx,y->py-x->ty));
    ab.push_back(std::hypot(x->px-y->tx,x->py-y->ty));
    bb.push_back(std::hypot(y->px-y->tx,y->py-y->ty));
  }
  std::cout<<"tolerance_px="<<tolerance<<" mutually_unique_pairs="<<pairs.size()
    <<" baseline_scored_unpaired="<<a.scored.size()-pairs.size()
    <<" candidate_scored_unpaired="<<b.scored.size()-pairs.size()<<'\n';
  for(const auto& [s,n]:a.statuses)std::cout<<"baseline "<<s<<'='<<n<<'\n';
  for(const auto& [s,n]:b.statuses)std::cout<<"candidate "<<s<<'='<<n<<'\n';
  summary("baseline_reference baseline",aa);summary("baseline_reference candidate",ba);
  summary("candidate_reference baseline",ab);summary("candidate_reference candidate",bb);
  std::cout<<"Matching uses both observed endpoints, never forecasts or local IDs. Both reference choices are reported.\n"
    <<"Nearby correspondence is not verified physical identity or ground truth; no unpaired sample is scored.\n";
  return int(pairs.size());
}
int main(int argc,char**argv){try{
  if(argc==2&&std::string(argv[1])=="--self-test"){
    const std::string header="frame,target_frame,id,status,origin_x,origin_y,predicted_x,predicted_y,observed_x,observed_y,error_px,hold_error_px,shadow_error_px\n";
    std::istringstream a(header+"1,4,1,scored,10,20,11,20,12,20,1,2,\n2,5,1,no_accepted_target,10,20,11,20,,,,,\n");
    std::istringstream b(header+"1,4,99,scored,10,20,12,20,12,20,0,2,\n1,4,1,scored,30,20,32,20,32,20,0,2,\n");
    const auto first=read(a),second=read(b);
    if(first.issued!=2||compare(first,second)!=1)throw std::runtime_error("paired sample matching failed");
    bool rejected=false;std::istringstream malformed(header+"1,4,1,scored,10,20,11,20,12,20,nan,2,\n");
    try{read(malformed);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("nonfinite metric accepted");
    std::istringstream shifted(header+"1,4,99,scored,10.5,20,13,20,12.5,20,0.5,2,\n");
    auto shift=read(shifted);auto pairs=nearby_pairs(first,shift,1);
    if(pairs.size()!=1||std::hypot(pairs[0].second->px-pairs[0].first->tx,pairs[0].second->py-pairs[0].first->ty)!=1)
      throw std::runtime_error("common reference not frozen");
    if(!nearby_pairs(first,shift,.1).empty())throw std::runtime_error("distance gate ignored");
    shift.scored.emplace("duplicate",shift.scored.begin()->second);
    if(!nearby_pairs(first,shift,1).empty()||!nearby_pairs(shift,first,1).empty())throw std::runtime_error("ambiguous pairs accepted");
    rejected=false;try{nearby_pairs(first,second,0);}catch(...){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid tolerance accepted");
    rejected=false;std::istringstream bad_coordinate(header+"1,4,1,scored,10,20,nan,20,12,20,1,2,\n");
    try{read(bad_coordinate);}catch(...){rejected=true;}
    if(!rejected)throw std::runtime_error("nonfinite coordinate accepted");
    return 0;
  }
  if(argc!=3&&argc!=5){std::cerr<<"usage: compare_predictions baseline.csv candidate.csv [--nearby-observations tolerance_px]\n";return 2;}
  std::ifstream a(argv[1]),b(argv[2]);if(!a||!b)throw std::runtime_error("cannot open audit CSV");
  if(argc==5){
    if(std::string(argv[3])!="--nearby-observations")throw std::runtime_error("unknown comparison option");
    if(!compare_nearby(read(a),read(b),std::stod(argv[4])))throw std::runtime_error("no mutually unique nearby observation pairs");
    return 0;
  }
  if(!compare(read(a),read(b)))throw std::runtime_error("no common scored observation pairs");
  std::cout<<"Only identical origin/target detector pairs compared; unmatched and unscored coverage is reported separately.\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
