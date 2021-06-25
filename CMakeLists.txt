cmake_minimum_required(VERSION 2.8)

project(cgtask3)

find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g -pthread")
#add_subdirectory(src/drv)

include_directories(src)

add_executable(main 
  src/main.cpp
  src/gpu/cmdqueue.cpp
  src/gpu/init.cpp
  src/lib/volk.c
)
target_link_libraries(main dl ${SDL2_LIBRARIES} ${Vulkan_LIBRARIES})