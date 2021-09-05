cmake_minimum_required(VERSION 2.8)

project(cgtask3)

find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g -pthread")
#add_subdirectory(src/drv)

include_directories(src)


set(spv-dir "src/lib/spirv-cross")

set(spirv-cross-sources
	${spv-dir}/spirv_cross.cpp
	${spv-dir}/spirv_parser.cpp
	${spv-dir}/spirv_cross_parsed_ir.cpp
	${spv-dir}/spirv_cfg.cpp
  ${spv-dir}/spirv_glsl.cpp)


add_executable(main 
  src/main.cpp
  src/gpu/vkerror.cpp
  src/gpu/driver.cpp
  src/gpu/resources.cpp
  src/gpu/shader.cpp
  src/lib/volk.c
  ${spirv-cross-sources}
)
target_link_libraries(main dl ${SDL2_LIBRARIES} ${Vulkan_LIBRARIES})