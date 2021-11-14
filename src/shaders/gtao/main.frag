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
#define SAMPLES 20
#define THIKNESS 0.05

#define UNIFORM_AO 0
#define COS_AO 1

#define AO_MODE COS_AO

#define PI 3.1415926535897932384626433832795

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

vec3 find_horizon_w0(in sampler2D depth, vec3 screen_start, vec2 dir, vec3 camera_pos, vec3 w0);

void main() {
  float frag_depth = texture(depth, screen_uv).r;
  if (frag_depth >= 1.f) {
    occlusion = 1.f;
    return;
  }
  
  vec3 view_vec = reconstruct_view_vec(screen_uv, frag_depth, fovy, aspect, znear, zfar);
  vec3 norm_view_vec = normalize(view_vec);
  
  vec3 w0 = -norm_view_vec;

  vec3 world_normal = decode_normal(texture(gbuffer_normal, screen_uv).xy);
  vec3 camera_normal = normalize((normal_mat * vec4(world_normal, 0)).xyz);

  float n_sign = -sign(view_vec.z);

  float sum = 0.f;
  const int USED_DIRS = DIR_COUNT;
  vec2 dir_radius = min(100.0/length(view_vec), 32.0) / vec2(textureSize(depth, 0));

  for (int dir_index = 0; dir_index < USED_DIRS; dir_index++) {    
    vec2 dir = dir_radius * DIRECTIONS[dir_index];
    vec3 end_pos = vec3(screen_uv, frag_depth) + vec3(DIRECTIONS[dir_index], 0);
    vec3 camera_end_pos = reconstruct_view_vec(end_pos.xy, end_pos.z, fovy, aspect, znear, zfar); 
    vec3 camera_dir = normalize(camera_end_pos - view_vec); //sample dir in camera space
    
    float nw = dot(camera_normal, w0);
    float nd = dot(camera_normal, camera_dir);

    vec3 nproj = nw * w0 + nd * camera_dir;
    float npoj_length = length(nproj);
    float n_dot_w0 = clamp(dot(normalize(nproj), w0), -1, 1);
    float n_angle = acos(n_dot_w0) * sign(nd);

    vec3 hor_pos = find_horizon_w0(depth, vec3(screen_uv, frag_depth), dir, view_vec, w0);

    
    vec3 delta = hor_pos - view_vec;
    float delta_length = length(delta);
    
    delta = (abs(delta_length) > 0.0001) ? (delta/delta_length) : vec3(0);
    float h_cos = clamp(dot(delta, w0), -1.0, 1.0);
    float h = acos(h_cos);

    //clamp h angle
    h = min(h + min(h - n_angle, PI/2), h);

  #if (AO_MODE == COS_AO)
    sum += npoj_length * 0.25 * max(-cos(2 * h - n_angle) + cos(n_angle) + 2*h*sin(n_angle), 0);  
  #endif
  #if (AO_MODE == UNIFORM_AO)
    sum += 1 - clamp(cos(h), 0, 1);
  #endif
  }

  sum = sum / (USED_DIRS);
  occlusion = sum;
}

vec3 find_horizon_w0(in sampler2D depth, vec3 screen_start, vec2 dir, vec3 camera_pos, vec3 w0) {

  vec3 prev_pos = camera_pos;
  float prev_depth = screen_start.z;
  
  float max_cos = -1.f;
  vec3 hor_pos = screen_start;

  for (int sample_index = 0; sample_index < SAMPLES; sample_index++) {
    vec2 offset = dir * (float(sample_index + 1)/SAMPLES);
    vec2 sample_pos = screen_start.xy + offset;
    float sampled_depth = textureLod(depth, sample_pos, 0).x;

    vec3 new_pos = reconstruct_view_vec(sample_pos.xy, sampled_depth, fovy, aspect, znear, zfar);

    float pos_delta = abs(prev_pos.z - new_pos.z);

    if (sampled_depth < prev_depth && pos_delta > 0.2) {
      break;
    }

    vec3 delta = new_pos - camera_pos;
    float delta_length = length(delta);
    delta = (abs(delta_length) > 0.001) ? (delta/delta_length) : vec3(0);
    float hcos = dot(delta, w0);
    
    if (hcos >= max_cos - 1e-6) {
      max_cos = hcos;
      hor_pos = new_pos;
    }

    prev_pos = new_pos;
    prev_depth = sampled_depth;
  }

  return hor_pos;
}