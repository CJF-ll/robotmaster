cmake_minimum_required(VERSION 3.16)
project(rm_vision_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(rm_vision_sim
  src/main.cpp
  src/config.cpp
  src/frame_source.cpp
  src/color_detector.cpp
  src/target_tracker.cpp
  src/result_writer.cpp
)

target_include_directories(rm_vision_sim PRIVATE include)
target_link_libraries(rm_vision_sim PRIVATE ${OpenCV_LIBS} Threads::Threads)
target_compile_options(rm_vision_sim PRIVATE -Wall -Wextra -Wpedantic)

add_executable(generate_test_video tools/generate_test_video.cpp)
target_link_libraries(generate_test_video PRIVATE ${OpenCV_LIBS})
target_compile_options(generate_test_video PRIVATE -Wall -Wextra -Wpedantic)
