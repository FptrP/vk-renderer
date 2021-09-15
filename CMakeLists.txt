cmake_minimum_required(VERSION 2.8)

project(cgtask3)

find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)
find_package(assimp REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -g -pthread")
#add_subdirectory(src/drv)

include_directories(src)


set(spv-dir "src/lib/spirv-cross")
set(imgui-dir "src/lib/imgui")

set(spirv-cross-sources
	${spv-dir}/spirv_cross.cpp
	${spv-dir}/spirv_parser.cpp
	${spv-dir}/spirv_cross_parsed_ir.cpp
	${spv-dir}/spirv_cfg.cpp
  ${spv-dir}/spirv_glsl.cpp)

set(imgui-sources
  ${imgui-dir}/imgui_draw.cpp
  ${imgui-dir}/imgui_impl_sdl.cpp
  ${imgui-dir}/imgui_impl_vulkan.cpp
  ${imgui-dir}/imgui_tables.cpp
  ${imgui-dir}/imgui_widgets.cpp
  ${imgui-dir}/imgui.cpp)

add_executable(main 
  src/rendergraph/resources.cpp
  src/rendergraph/rendergraph.cpp
  src/rendergraph/gpu_ctx.cpp
  src/main.cpp
  src/backbuffer_subpass2.cpp
  src/gbuffer_subpass.cpp
  src/gpu/vkerror.cpp
  src/gpu/driver.cpp
  src/gpu/resources.cpp
  src/gpu/shader.cpp
  src/framegraph/framegraph.cpp
  src/scene/scene.cpp
  src/lib/volk.c
  ${spirv-cross-sources}
  ${imgui-sources}
)
target_link_libraries(main dl ${SDL2_LIBRARIES} ${Vulkan_LIBRARIES} assimp)