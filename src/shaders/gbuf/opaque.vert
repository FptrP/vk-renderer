#version 460 core

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_norm;
layout (location = 2) in vec2 in_uv;

layout (set = 0, binding = 0) uniform GbufConst {
  mat4 camera_projection;
  mat4 camera_normal;
};

struct Transform {
  mat4 model;
  mat4 normal;
};

layout (std430, set = 0, binding = 1) readonly buffer TransformBuffer {
  Transform transforms[];
};

layout (location = 0) out vec3 out_normal;
layout (location = 1) out vec2 out_uv;

layout (push_constant) uniform push_data {
  uint transform_index;
  uint albedo_index;
  uint mr_index;
};

void main() {
  out_normal = normalize((camera_normal * transforms[transform_index].normal * vec4(in_norm, 0)).xyz);
  //mat4 norm_mat = transpose(inverse(camera_projection * transforms[transform_index].model));
  //out_normal = normalize((norm_mat * vec4(in_norm, 0)).xyz);
  out_uv = in_uv;
  gl_Position = camera_projection * transforms[transform_index].model * vec4(in_pos, 1);
}