cmake_minimum_required(VERSION 3.0.2)
project(imu_gtsam)

add_compile_options(-std=c++17)

find_package( Boost COMPONENTS program_options REQUIRED )

set(CMAKE_CXX_FLAGS "-O3 -g")

include_directories(
  /usr/include/eigen3/
  /usr/local/include
)

add_executable(${PROJECT_NAME}_main main.cpp)


target_link_libraries(${PROJECT_NAME}_main ${Boost_LIBRARIES} gtsam tbb)


