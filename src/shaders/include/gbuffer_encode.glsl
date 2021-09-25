#ifndef GBUFFER_ENCODE_GLSL_INCLUDED
#define GBUFFER_ENCODE_GLSL_INCLUDED


float sign_nz(in float k) {
  return (k >= 0.0) ? 1.0 : -1.0;
}

vec2 sign_nz(in vec2 v) {
  return vec2(sign_nz(v.x), sign_nz(v.y));
}

vec3 sign_nz(in vec3 v) {
  return vec3(sign_nz(v.x), sign_nz(v.y), sign_nz(v.z));
}

vec2 encode_normal(in vec3 v) 
{
  float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
  vec2 result = v.xy * (1.0 / l1norm);
  
  if (v.z < 0.0) {
    result = (1.0 - abs(result.yx)) * sign_nz(result.xy);
  }

  return 0.5f * result + vec2(0.5f, 0.5f);
}

vec3 decode_normal(in vec2 uv)
{
  uv = 2.f * (uv - vec2(0.5f, 0.5f));
  vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
  if (v.z < 0.0) {
    v.xy = (1.0 - abs(v.yx)) * sign_nz(v.xy);
  }

  return normalize(v);
}

vec3 sample_gbuffer_normal(in sampler2D normal_tex, in vec2 uv)
{
  vec2 t = texture(normal_tex, uv).xy;
  return decode_normal(t); 
}


float linearize_depth(float d, float zNear,float zFar)
{
  return zNear * zFar / (zFar + d * (zNear - zFar));
}

float linearize_depth2(float d, float n, float f)
{
  return n * f / (d * (f - n) - f);
}

vec3 reconstruct_view_vec(vec2 uv, float d, float fovy, float aspect, float z_near, float z_far)
{
  float tg_alpha = tan(fovy/2);
  float z = linearize_depth2(d, z_near, z_far);

  float xd = 2 * uv.x - 1;
  float yd = 2 * uv.y - 1;

  float x = -(xd) * (z * aspect * tg_alpha);
  float y = -(yd) * (z * tg_alpha);
  return vec3(x, y, z);
}

#endif