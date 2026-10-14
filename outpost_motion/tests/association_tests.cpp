#include "association.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){
  using outpost::associate;const double inf=std::numeric_limits<double>::infinity();
  auto a=associate({{1,2},{2,inf}},2,15,.1);
  require(a.assignment==std::vector<int>({1,0}),"greedy candidate stealing remains");
  a=associate({{1,1},{1,1}},2,15,.1);
  require(a.assignment==std::vector<int>({-1,-1})&&a.ambiguous_detections[0]&&a.ambiguous_detections[1],
          "equal global solutions forced into identities");
  a=associate({{0,10,10},{10,0,10},{10,10,0}},3,15,1);
  require(a.assignment==std::vector<int>({0,1,2}),"poor competing edges suppressed strong assignment");
  a=associate({{1},{1},{inf}},1,15,.1);
  require(a.assignment==std::vector<int>({-1,-1,-1})&&a.ambiguous_detections[0],"one detection reused or ambiguity born anew");
  require(associate({},3,15,1).ambiguous_detections.size()==3,"empty track set lost detection cardinality");
  require(associate({{},{}},0,15,1).assignment==std::vector<int>({-1,-1}),"no detections not handled");
  require(associate({{20}},1,15,1).assignment[0]==-1,"unmatched option not honored");
  bool rejected=false;try{associate({{std::numeric_limits<double>::quiet_NaN()}},1,15,1);}catch(const std::exception&){rejected=true;}
  require(rejected,"NaN cost accepted");
  // Exhaustive oracle on deterministic small random problems, including gates.
  std::mt19937 random(2147);
  for(int sample=0;sample<300;++sample){
    const int n=1+random()%4,m=1+random()%4;
    std::vector<std::vector<double>> costs(n,std::vector<double>(m));
    for(auto& row:costs)for(auto& c:row)c=random()%5==0?inf:double(random()%25);
    double optimum=inf;std::vector<bool> used(m,false);
    std::vector<int> assignment(n,-1);
    std::vector<std::pair<double,std::vector<int>>> enumerated;
    std::function<void(int,double)> enumerate=[&](int row,double total){
      if(row==n){optimum=std::min(optimum,total);enumerated.push_back({total,assignment});return;}
      assignment[row]=-1;enumerate(row+1,total+15);
      for(int j=0;j<m;++j)if(!used[j]&&std::isfinite(costs[row][j])){
        assignment[row]=j;used[j]=true;enumerate(row+1,total+costs[row][j]);used[j]=false;
      }
    };
    const double margin=sample%2?.1:1.1;
    enumerate(0,0);const auto answer=associate(costs,m,15,margin);
    require(std::abs(answer.optimal_cost-optimum)<1e-8,"assignment differs from exhaustive optimum");
    std::fill(used.begin(),used.end(),false);
    for(int j:answer.assignment)if(j>=0){require(!used[j],"assignment is not one-to-one");used[j]=true;}
    for(int i=0;i<n;++i)if(answer.assignment[i]>=0)
      for(const auto& alternative:enumerated)if(alternative.first<=optimum+margin+1e-10)
        require(alternative.second[i]==answer.assignment[i],"accepted edge changes in near-optimal solution");
  }
  std::cout<<"global assignment, ambiguity, unmatched and exhaustive optimum checks passed\n";
}
