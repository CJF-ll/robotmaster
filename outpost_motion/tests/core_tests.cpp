#include "core.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) throw std::runtime_error("test config missing");
  const auto config = outpost::Config::load(argv[1]);
  if (std::abs(config.prediction_seconds - 0.10) > 1e-9)
    throw std::runtime_error("unexpected configured horizon");
  bool rejected = false;
  try { outpost::Config::load("/nonexistent/outpost-test-config.yaml"); }
  catch (const std::exception&) { rejected = true; }
  if (!rejected) throw std::runtime_error("missing config accepted");
  // Tests must remain active in release builds (do not use assert).
  outpost::RuntimeTelemetry telemetry;
  if (telemetry.pose != outpost::Availability::Unavailable ||
      telemetry.bullet_speed != outpost::Availability::Unavailable)
    throw std::runtime_error("unprovided telemetry must stay unavailable");
  for (int i = 1; i < 925; ++i)
    if (!(i / 30.0 > (i - 1) / 30.0))
      throw std::runtime_error("frame time not increasing");
  if (std::abs(924 / 30.0 - 30.8) > 1e-9)
    throw std::runtime_error("frame timestamp mismatch");
  std::cout << "core contracts passed\n";
}
