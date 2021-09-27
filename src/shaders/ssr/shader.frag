#version 460
#include <gbuffer_encode.glsl>

layout (location = 0) in vec2 screen_uv;

layout (set = 0, binding = 0) uniform sampler2D normal_tex;
layout (set = 0, binding = 1) uniform sampler2D depth_tex;
layout (set = 0, binding = 2) uniform sampler2D frame_tex;

layout (set = 0, binding = 3) uniform SSRParams {
  mat4 camera_normal;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

layout (location = 0) out vec4 out_reflection;

void main() {
  float pixel_depth = texture(depth_tex, screen_uv).x;
  vec3 pixel_normal_world = sample_gbuffer_normal(normal_tex, screen_uv);
  vec3 pixel_normal = normalize((camera_normal * vec4(pixel_normal_world, 0)).xyz);
  
  vec3 view_vec = reconstruct_view_vec(screen_uv, pixel_depth, fovy, aspect, znear, zfar);

  vec3 R = reflect(view_vec, pixel_normal);

  vec3 start = vec3(screen_uv, pixel_depth);
  vec3 p = project_view_vec(view_vec + R, fovy, aspect, znear, zfar);
  vec3 delta = normalize(p - start);

  if (abs(delta.z) < 0.0000001) {
    out_reflection = vec4(0, 0, 0, 0);
    return;  
  }

  float t_bound = (1-start.z)/delta.z;
  float u_bound = max((1 - start.x)/delta.x, -start.x/delta.x);
  float v_bound = max((1 - start.y)/delta.y, -start.y/delta.y); 
  t_bound = min(t_bound, min(u_bound, v_bound));

  vec3 end = start + t_bound * delta;
  delta = end - start;

  ivec2 pixel_dist = ivec2(textureSize(frame_tex, 0).xy * abs(delta.xy));

  int steps = max(pixel_dist.x, pixel_dist.y);
  vec3 vec_step = delta/steps;
  steps = clamp(steps, 1, 500);

  for (int i = 0; i < steps - 1; i++) {
    vec3 p = start + vec_step * i;

    float depth = texture(depth_tex, p.xy).x;

    if (p.z - 0.0001 > depth) {
      out_reflection = texture(frame_tex, p.xy);
      return;
    }
  }


  out_reflection = vec4(0, 0, 0, 0);
}