#include "association.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace outpost {
namespace {
struct Solution {std::vector<int> assignments;double cost=0;};
// Rectangular shortest augmenting-path assignment with explicit dummy columns.
Solution solve(const std::vector<std::vector<double>>& costs,size_t detections,
               double miss_cost,int forbidden_row=-1,int forbidden_column=-1){
  const size_t n=costs.size(),m=detections+n;
  std::vector<double> u(n+1),v(m+1);
  std::vector<size_t> owner(m+1),previous(m+1);
  auto cost=[&](size_t row,size_t col){
    if(col>=detections)return miss_cost;
    if(int(row)==forbidden_row&&int(col)==forbidden_column)return std::numeric_limits<double>::infinity();
    return costs[row][col];
  };
  for(size_t row=1;row<=n;++row){
    owner[0]=row;size_t column=0;
    std::vector<double> remaining(m+1,std::numeric_limits<double>::infinity());
    std::vector<bool> visited(m+1,false);
    do{
      visited[column]=true;const auto current=owner[column];
      double delta=std::numeric_limits<double>::infinity();size_t next=0;
      for(size_t j=1;j<=m;++j)if(!visited[j]){
        const double reduced=cost(current-1,j-1)-u[current]-v[j];
        if(reduced<remaining[j]){remaining[j]=reduced;previous[j]=column;}
        if(remaining[j]<delta){delta=remaining[j];next=j;}
      }
      if(!std::isfinite(delta))throw std::runtime_error("assignment dummy feasibility lost");
      for(size_t j=0;j<=m;++j){
        if(visited[j]){u[owner[j]]+=delta;v[j]-=delta;}
        else remaining[j]-=delta;
      }
      column=next;
    }while(owner[column]!=0);
    do{const size_t back=previous[column];owner[column]=owner[back];column=back;}while(column);
  }
  Solution result;result.assignments.assign(n,-1);
  for(size_t col=1;col<=m;++col)if(owner[col]&&col<=detections)
    result.assignments[owner[col]-1]=int(col-1);
  for(size_t i=0;i<n;++i)result.cost+=result.assignments[i]<0?miss_cost:costs[i][result.assignments[i]];
  return result;
}
}
AssociationResult associate(const std::vector<std::vector<double>>& costs,size_t detections,
                            double miss_cost,double ambiguity_margin){
  if(!std::isfinite(miss_cost)||miss_cost<=0||!std::isfinite(ambiguity_margin)||ambiguity_margin<0)
    throw std::runtime_error("invalid assignment policy");
  for(const auto& row:costs){
    if(row.size()!=detections)throw std::runtime_error("ragged assignment matrix");
    for(double cost:row)if(std::isnan(cost)||cost<0)throw std::runtime_error("invalid assignment cost");
  }
  const auto best=solve(costs,detections,miss_cost);
  AssociationResult result;result.assignment=best.assignments;result.optimal_cost=best.cost;
  result.ambiguous_tracks.assign(costs.size(),false);result.ambiguous_detections.assign(detections,false);
  for(size_t row=0;row<costs.size();++row)if(best.assignments[row]>=0){
    // Excluding each chosen edge tests whether another near-optimal COMPLETE
    // assignment changes ownership. Dummy-column permutations are irrelevant.
    const auto alternative=solve(costs,detections,miss_cost,int(row),best.assignments[row]);
    if(alternative.cost-best.cost>ambiguity_margin+1e-10)continue;
    for(size_t i=0;i<costs.size();++i)if(alternative.assignments[i]!=best.assignments[i]){
      result.ambiguous_tracks[i]=true;
      for(int detection:{best.assignments[i],alternative.assignments[i]})
        if(detection>=0)result.ambiguous_detections[detection]=true;
    }
  }
  for(size_t i=0;i<costs.size();++i)if(result.ambiguous_tracks[i])result.assignment[i]=-1;
  return result;
}
}
