#version 450

layout(location = 0) in vec4 v_Color;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec3 v_HalfDir;
layout(location = 0) out vec4 o_Target;

void main() {
    float diffuse = max(0.0, dot(normalize(v_Normal), normalize(v_HalfDir)));
    o_Target = diffuse * v_Color;
}
