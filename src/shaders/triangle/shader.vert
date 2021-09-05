#version 460 core

vec2 verts[] = vec2[](
  vec2(-1, 1),
  vec2(1, 1),
  vec2(0, -1)
);

void main() {
  gl_Position = vec4(verts[gl_VertexIndex], 0.1, 1);
}
