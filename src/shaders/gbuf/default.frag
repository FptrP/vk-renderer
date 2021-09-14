#version 460 core

layout (location = 0) in vec3 in_normal;

layout (location = 0) out vec4 out_albedo;
layout (location = 1) out vec4 out_normal;
layout (location = 2) out vec4 out_material;

void main() {
  out_albedo = vec4(0, 0, 0, 0);
  out_normal = vec4(in_normal, 0);
  out_material = vec4(0, 0, 0, 0);
}