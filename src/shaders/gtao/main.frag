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

layout (push_constant) uniform PushConstants {
  float angle_offset;
};

#define RADIUS 0.01
#define SAMPLES 20
#define THIKNESS 0.05

#define UNIFORM_AO 0
#define COS_AO 1

#define AO_MODE COS_AO

#define PI 3.1415926535897932384626433832795

float gtao_direction(in ivec2 pos);
float gtao_normal_space(in ivec2 pos, in vec2 screen_uv);
float gtao_camera_space(in ivec2 pos, in vec2 screen_uv, uint dirs_count);

vec3 find_horizon_w0(in sampler2D depth, vec3 screen_start, vec2 dir, vec3 camera_pos, vec3 w0);

void main() {
  ivec2 pixel_pos = ivec2(gl_FragCoord.xy);
  occlusion = gtao_camera_space(pixel_pos, screen_uv, 1);
}

float gtao_normal_space(in ivec2 pos, in vec2 screen_uv) {
  return 0.f; 
}

float gtao_camera_space(in ivec2 pos, in vec2 screen_uv, uint dirs_count) {
  float frag_depth = texture(depth, screen_uv).r;
  if (frag_depth >= 1.f) {
    return 1.f;
  }

  vec3 camera_pos = reconstruct_view_vec(screen_uv, frag_depth, fovy, aspect, znear, zfar);
  vec3 w0 = -normalize(camera_pos);
  vec3 camera_normal = normalize((normal_mat * vec4(decode_normal(texture(gbuffer_normal, screen_uv).xy), 0)).xyz);

  vec2 dir_radius = min(100.0/length(camera_pos), 32.0) / vec2(textureSize(depth, 0));
  float base_angle = gtao_direction(pos) + angle_offset; 
  float sum = 0.f;

  for (int dir_index = 0; dir_index < dirs_count; dir_index++) {
    float angle = 2 * PI * (base_angle + float(dir_index)/float(dirs_count));

    vec2 sample_direction = dir_radius * vec2(cos(angle), sin(angle));
    vec3 sample_end_pos = reconstruct_view_vec(screen_uv + sample_direction, frag_depth, fovy, aspect, znear, zfar);

    vec3 slice_normal = normalize(cross(w0, -sample_end_pos));
    vec3 normal_projected = camera_normal - dot(camera_normal, slice_normal) * slice_normal;
    float n = PI/2.0 - acos(dot(normalize(normal_projected), normalize(sample_end_pos - camera_pos)));

    float h_cos = -1.0;

    for (int i = 1; i <= SAMPLES; i++) {
      vec2 tc = screen_uv + (float(i)/SAMPLES) * sample_direction;
      float sample_depth = textureLod(depth, tc, 0).r;
      
      vec3 sample_offset = reconstruct_view_vec(tc, sample_depth, fovy, aspect, znear, zfar) - camera_pos;
      
      if (length(sample_offset) > 1.0) {
        break;
      }
      
      float sample_cos = dot(w0, normalize(sample_offset));

      if (sample_cos > h_cos) {
        h_cos = sample_cos;
      }
    }

    //h_cos = clamp(h_cos, -1.0 + 1e-3, 1.0 - 1e-3);
    float h = acos(h_cos);
    h = min(n + min(h - n, PI/2.0), h);
    sum += length(normal_projected) * 0.25 * max(-cos(2 * h - n) + cos(n) + 2*h*sin(n), 0);
  }

  return 2 * sum/float(dirs_count);
}

float gtao_direction(in ivec2 pos) {
  return (1.0 / 16.0) * ((((pos.x + pos.y) & 3) << 2) + (pos.x & 3));
}
