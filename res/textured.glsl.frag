#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 1, binding = 0) uniform sampler2D texture_addresses[];

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec2 fragUv;
layout(location = 1) in flat uint texIndex;


void main() {
    
    outColor = vec4( texture(texture_addresses[texIndex], fragUv).rgb, 1.0f);
}