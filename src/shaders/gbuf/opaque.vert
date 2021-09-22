#version 460 core

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_norm;
layout (location = 2) in vec2 in_uv;

layout (set = 0, binding = 0) uniform GbufConst {
  mat4 camera_projection;
};

layout (location = 0) out vec3 out_normal;
layout (location = 1) out vec2 out_uv;

layout (push_constant) uniform push_data {
  mat4 model;
};

void main() {
  out_normal = in_norm;
  out_uv = in_uv;
  gl_Position = camera_projection * model * vec4(in_pos, 1);
}