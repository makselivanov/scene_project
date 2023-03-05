#pragma once

#include <GL/glew.h>

const char vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4x3 bones[64];
uniform int is_rigged;
uniform int is_instance;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;
layout (location = 3) in ivec4 in_joints;
layout (location = 4) in vec4 in_weights;
layout (location = 5) in vec3 instance; //TODO add
uniform mat4 instance_turn;

out vec3 normal;
out vec2 texcoord;
out vec4 weights;
out vec3 position;

void main()
{
    mat4x3 average = mat4x3(0);
    float sum = 0;
    for (int i = 0; i < 4; ++i) {        //was 4
        sum += in_weights[i];
        average += in_weights[i] * bones[in_joints[i]];
    }
    average /= sum;
    vec3 new_instance;
    mat4 new_instance_turn;
    if (is_instance == 0) {
        new_instance = vec3(0);
        new_instance_turn = mat4(1);
    } else {
        new_instance = instance;
        new_instance_turn = instance_turn;
    }
    mat4 shift_view = mat4( 1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            new_instance.x, new_instance.y, new_instance.z, 1 );
    mat4 inv_shift_view = mat4( 1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0,
                                -new_instance.x, -new_instance.y, -new_instance.z, 1 );
    mat4 new_view = view * shift_view * new_instance_turn * inv_shift_view;
    if (is_rigged != 0) {
        gl_Position = projection * new_view * model * mat4(average) * vec4(in_position + new_instance, 1.0);
        normal = mat3(model) * mat3(average) * in_normal;
    } else {
        gl_Position = projection * new_view * model * vec4(in_position + new_instance, 1.0);
        normal = mat3(model) * in_normal;
    }
    position = (model * vec4(in_position, 1.0)).xyz;
    weights = in_weights;
    texcoord = in_texcoord;
}
)";

const char fragment_shader_source[] =
        R"(#version 330 core

uniform sampler2D albedo;
uniform vec4 color;
uniform int use_texture;
uniform vec3 camera_position;

//uniform float metallic;
uniform float roughness;

in vec3 normal;
in vec3 position;
//uniform vec3 point_light_position;
//uniform vec3 point_light_color;
//uniform vec3 point_light_attenuation;


uniform vec3 light_direction;

layout (location = 0) out vec4 out_color;

in vec4 weights;
in vec2 texcoord;

void main()
{
    vec4 albedo_color;

    if (use_texture == 1)
        albedo_color = texture(albedo, texcoord);
    else
        albedo_color = color;

    float ambient = 0.4; //0.4
    float cosine = dot(normal, light_direction);
    float light_factor = max(0.0, dot(normalize(normal), light_direction));
    vec3 reflected_direction = 2.0 * normal * cosine - light_direction;
    vec3 view_direction = normalize(camera_position - position);
    float specular_power = 1 / (roughness * roughness) - 1;
    //vec3 sun = (diffuse(light_direction, albedo_color) + specular(light_direction, albedo_color)) * vec3(1.f, 0.9f, 0.8f);
    //float r = distance(position, point_light_position);
    //vec3 dir = normalize(point_light_position - position);
    //vec3 source = (diffuse(dir, albedo_color) + specular(dir_albedo_color, albedo_color)) * point_light_color / dot(point_light_attenuation, vec3(1, r, r*r));
    float specular_factor = pow(max(0.0, dot(reflected_direction, view_direction)),
                                specular_power);

    out_color = vec4(albedo_color.rgb * (ambient + light_factor + specular_factor), albedo_color.a);
}
)";

const char vertex_shader_env_source[] = R"(#version 330 core

const vec2 VERTICES[6] = vec2[6](
	vec2(-1.0, 1.0),
	vec2(-1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, -1.0)
);

uniform mat4 view;
uniform mat4 projection;

out vec3 position;

void main() {
    vec2 vertex = VERTICES[gl_VertexID];
    vec4 ndc = vec4(vertex, 0.0, 1.0);
    mat4 view_projection_inverse = inverse(projection * view);

    vec4 clip_space = view_projection_inverse * ndc;
    position = clip_space.xyz / clip_space.w;
    gl_Position = vec4(vertex, 0.99999, 1.);
    //gl_Position = projection * view * vec4(position, 1.0);
}
)";

const char fragment_shader_env_source[] = R"(#version 330 core

uniform vec3 camera_position;
uniform sampler2D environment_texture;
in vec3 position;
layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main()
{
    vec3 dir = position - camera_position;
    float x = atan(dir.z, dir.x) / PI * 0.5 + 0.5;
    float y = -atan(dir.y, length(dir.xz)) / PI * 0.5 + 0.5;
    out_color = vec4(texture(environment_texture, vec2(x, y)).rgb, 1.0);
}
)";

const char shadow_vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 model;
uniform mat4 transform;

layout (location = 0) in vec3 in_position;

void main()
{
    gl_Position = transform * model * vec4(in_position, 1.0);
}
)";

const char shadow_fragment_shader_source[] =
        R"(#version 330 core

void main()
{}
)";

const char vertex_shader_source_simple[] =
        R"(#version 330 core
uniform mat4 view;
uniform mat4 projection;
uniform vec3 bbox_min;
uniform vec3 bbox_max;
layout (location = 0) in vec3 in_position;
out vec3 position;
void main()
{
    position = bbox_min + in_position * (bbox_max - bbox_min);
    gl_Position = projection * view * vec4(position, 1.0);
}
)";

const char fragment_shader_source_simple[] =
        R"(#version 330 core
uniform vec3 camera_position;
uniform vec3 light_direction;
uniform vec3 bbox_min;
uniform vec3 bbox_max;
layout (location = 0) out vec4 out_color;
const float PI = 3.1415926535;
in vec3 position;
void main()
{
    out_color = vec4(0.5, .5, 1, 1.0);
}
)";

const char msdf_vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 transform;
layout (location = 0) in vec2 in_position;
layout (location = 1) in vec2 in_texcoord;

out vec2 texcoord;

void main()
{
    texcoord = in_texcoord;
    gl_Position = transform * vec4(in_position, 0.0, 1.0);
}
)";

const char msdf_fragment_shader_source[] =
        R"(#version 330 core

in vec2 texcoord;
layout (location = 0) out vec4 out_color;
uniform float sdf_scale;
uniform sampler2D sdf_texture;

float median(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main()
{
    float texture_value = median(texture(sdf_texture, texcoord).rgb);
    float sdf = (texture_value - .5) * sdf_scale;
    float alpha = smoothstep(-0.5, 0.5, sdf);
    out_color = vec4(1., 1., 1., alpha);
}
)";


GLuint create_shader(GLenum type, const char * source);
