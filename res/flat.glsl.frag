#version 450

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec3 fragColor;

layout(set = 1, binding = 0) uniform sampler2D texture_addresses[];

void main() {
    
    outColor = vec4(fragColor, 1.0f);
}