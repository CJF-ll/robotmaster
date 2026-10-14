#pragma once
#include "tracker.hpp"
namespace outpost {
struct VisibilityConfig {
  double side_distance_ratio=.25;
  double angle_degrees=25;
  double length_ratio=2;
  double min_axial_overlap=.6;
};
enum class Evidence { AcceptedPair, CompatibleSingleLamp, None };
struct RenderDecision {
  Evidence evidence=Evidence::None;
  bool current=false, future=false;
};
// A compatible lamp is geometric corroboration, not a full observation and
// not proof of physical identity. This policy NEVER updates filter state.
RenderDecision visibility(const TrackOutput&, const std::vector<Lightbar>&,
                          const VisibilityConfig&);
// Pass Detector::pair_candidates(), NOT only accepted detector outputs: lamps
// involved in unresolved pairings cannot corroborate a different coast track.
std::vector<RenderDecision> visibility_batch(const std::vector<TrackOutput>&,
    const std::vector<Lightbar>&, const std::vector<Detection>&, const VisibilityConfig&);
}
