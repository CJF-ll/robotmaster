#include <exception>
#include <iostream>
#include <thread>

#include "bounded_queue.hpp"
#include "color_detector.hpp"
#include "config.hpp"
#include "error_state.hpp"
#include "frame_source.hpp"
#include "result_writer.hpp"
#include "target_tracker.hpp"
#include "types.hpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <config-file>\n";
    return 1;
  }

  try {
    const Config config = load_config(argv[1]);
    BoundedQueue<ImageFrame> frame_queue(config.queue_capacity);
    BoundedQueue<DetectionPacket> observation_queue(config.queue_capacity);
    BoundedQueue<OutputPacket> result_queue(config.queue_capacity);
    ErrorState errors;

    FrameSource source(config, frame_queue, errors);
    ColorDetector detector(config, frame_queue, observation_queue, errors);
    TargetTracker tracker(config, observation_queue, result_queue, errors);
    ResultWriter writer(config, result_queue, errors);

    std::thread source_thread(&FrameSource::run, &source);
    std::thread detector_thread(&ColorDetector::run, &detector);
    std::thread tracker_thread(&TargetTracker::run, &tracker);
    std::thread writer_thread(&ResultWriter::run, &writer);

    source_thread.join();
    detector_thread.join();
    tracker_thread.join();
    writer_thread.join();

    errors.rethrow_if_set();
    std::cout << "Vision pipeline completed successfully.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
