#version 460
#include <gbuffer_encode.glsl>

layout (location = 0) in vec2 screen_uv;
layout (location = 0) out float occlusion;

layout (set = 0, binding = 0) uniform sampler2D depth;

layout (set = 0, binding = 1) uniform GTAOParams {
  mat4 normal_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

layout (set = 0, binding = 2) uniform sampler2D gbuffer_normal;

#define DIR_COUNT 8
#define RADIUS 0.01
#define SAMPLES 10
#define THIKNESS 0.05

const vec2 DIRECTIONS[DIR_COUNT] = vec2[DIR_COUNT](
  normalize(vec2(1, 0)),
  normalize(vec2(1, -1)),
  normalize(vec2(0, -1)),
  normalize(vec2(-1, -1)),
  normalize(vec2(-1, 0)),
  normalize(vec2(-1, 1)),
  normalize(vec2(0, 1)),
  normalize(vec2(1, 1))
);

vec3 find_horizon(in sampler2D depth, vec3 screen_start, vec2 dir, uint samples, in vec2 znear_zfar);

void main() {
  float frag_depth = texture(depth, screen_uv).r;
  if (frag_depth >= 1.f) {
    occlusion = 1.f;
    return;
  }
  
  vec3 view_vec = reconstruct_view_vec(screen_uv, frag_depth, fovy, aspect, znear, zfar);
  vec3 norm_view_vec = normalize(view_vec);
  vec3 world_normal = decode_normal(texture(gbuffer_normal, screen_uv).xy);
  vec3 camera_normal = (normal_mat * vec4(world_normal, 0)).xyz;

  float n_sign = -sign(view_vec.z);

  float sum = 0.f;

  for (int dir_index = 0; dir_index < DIR_COUNT; dir_index++) {    
    vec2 dir = RADIUS * DIRECTIONS[dir_index];
    vec3 hor_vec = find_horizon(depth, vec3(screen_uv, frag_depth), dir, SAMPLES, vec2(znear, zfar));
    vec3 camera_vec = reconstruct_view_vec(hor_vec.xy, hor_vec.z, fovy, aspect, znear, zfar);
    vec3 delta = normalize(camera_vec - view_vec);
    //float min_cos_h = clamp(dot(delta, -norm_view_vec), 0, 1);
    float min_cos_h = n_sign * delta.z;
    float h = acos(min_cos_h);
    float half_arc = 1 - min_cos_h;//0.25 * (1 - cos(2 * h));
    sum += half_arc;
  }

  sum = sum / (DIR_COUNT);
  occlusion = sum;
}

vec3 find_horizon(in sampler2D depth, vec3 screen_start, vec2 dir, uint samples, in vec2 znear_zfar) {
  float prev_z = linearize_depth2(screen_start.z, znear_zfar.x, znear_zfar.y);
  float prev_depth = screen_start.z;

  vec3 hor_vec = screen_start;

  for (int sample_index = 0; sample_index < samples; sample_index++) {
    vec2 offset = dir * (float(sample_index + 1)/samples);
    vec2 sample_pos = screen_start.xy + offset;
    float sampled_depth = textureLod(depth, sample_pos, 0).x;   
    sampled_depth = min(sampled_depth, screen_start.z);

    float sampled_z = linearize_depth2(sampled_depth, znear_zfar.x, znear_zfar.y);
    if (sampled_depth < prev_depth && abs(prev_z - sampled_z) > 0.07) {
      sampled_depth = prev_depth;
      sampled_z = prev_z;
    }

    if (sampled_depth < hor_vec.z) {
      hor_vec = vec3(sample_pos, sampled_depth);
    }

    prev_z = sampled_z;
    prev_depth = sampled_depth;
  }

  return hor_vec;
}