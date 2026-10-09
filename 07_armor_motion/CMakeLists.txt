cmake_minimum_required(VERSION 3.16)
project(outpost_motion_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED)

add_library(outpost_core
  src/config.cpp
  src/armor_detector.cpp
  src/rigid_model.cpp
)
target_include_directories(outpost_core PUBLIC include)
target_link_libraries(outpost_core PUBLIC ${OpenCV_LIBS})
target_compile_options(outpost_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(outpost_solver src/main.cpp)
target_link_libraries(outpost_solver PRIVATE outpost_core)
target_compile_options(outpost_solver PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  add_executable(outpost_tests tests/test_main.cpp)
  target_link_libraries(outpost_tests PRIVATE outpost_core)
  target_compile_options(outpost_tests PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_definitions(outpost_tests PRIVATE
    OUTPOST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME outpost_tests COMMAND outpost_tests)
endif()
