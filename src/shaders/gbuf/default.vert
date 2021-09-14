#version 460 core

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_norm;
layout (location = 2) in vec2 in_uv;

layout (set = 0, binding = 0) uniform GbufConst {
  mat4 mvp;
};

layout (location = 0) out vec3 out_normal;

void main() {
  gl_Position = mvp * vec4(in_pos, 1);
  out_normal = in_norm;
}