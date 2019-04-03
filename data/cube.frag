#version 450

layout(location = 0) in vec3 v_Normal;
layout(location = 1) in vec3 v_HalfDir;
layout(location = 0) out vec4 o_Target;

const vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

void main() {
    float diffuse = max(0.0, dot(normalize(v_Normal), normalize(v_HalfDir)));
    o_Target = diffuse * color;
}
