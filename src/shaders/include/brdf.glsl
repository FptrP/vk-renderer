#ifndef BRDF_GLSL_INCLUDED
#define BRDF_GLSL_INCLUDED

const float PI = 3.1415926535897932384626433832795;

vec3 fresnelSchlick(float cos_theta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}  

vec3 F0_approximation(vec3 albedo, float metallic) {
  vec3 F0 = vec3(0.04);
  return mix(F0, albedo, metallic);
}
#if 0

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
  float a      = roughness*roughness;
  float a2     = a*a;
  float NdotH  = max(dot(N, H), 0.0);
  float NdotH2 = NdotH*NdotH;
	
  float num   = a2;
  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;
	
  return num / denom;
}
#else

float DistributionGGX(vec3 N, vec3 H, float alpha)
{
  float NoH = dot(N, H);
  float alpha2 = alpha * alpha;
  float NoH2 = NoH * NoH;
  float den = NoH2 * alpha2 + (1 - NoH2);
  return (((NoH2 > 0)? 1 : 0) * alpha2) / ( PI * den * den );
}

#endif


float GeometrySchlickGGX(float NdotV, float roughness)
{
  float r = (roughness + 1.0);
  float k = (r*r) / 8.0;

  float num   = NdotV;
  float denom = NdotV * (1.0 - k) + k;
  return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2  = GeometrySchlickGGX(NdotV, roughness);
  float ggx1  = GeometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}

#endif