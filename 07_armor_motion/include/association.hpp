#pragma once
#include <cstddef>
#include <vector>
namespace outpost {
struct AssociationResult {
  std::vector<int> assignment;
  std::vector<bool> ambiguous_tracks,ambiguous_detections;
  double optimal_cost=0;
};
// Infinite costs are gated-out edges. Every track may remain unmatched at
// miss_cost; every real detection can be assigned at most once.
AssociationResult associate(const std::vector<std::vector<double>>& costs,
                            std::size_t detections,double miss_cost,double ambiguity_margin);
}
