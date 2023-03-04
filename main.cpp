#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>
#include <string_view>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <random>
#include <map>
#include <cmath>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include "gltf_loader.hpp"
#include "stb_image.h"

#define DEBUG

std::string to_string(std::string_view str)
{
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4x3 bones[64];
uniform int is_rigged;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;
layout (location = 3) in ivec4 in_joints;
layout (location = 4) in vec4 in_weights;

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
    if (is_rigged != 0) {
        gl_Position = projection * view * model * mat4(average) * vec4(in_position, 1.0);
        normal = mat3(model) * mat3(average) * in_normal;
    } else {
        gl_Position = projection * view * model * vec4(in_position, 1.0);
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

//uniform float glossiness;
//uniform float roughness;

in vec3 normal;
in vec3 position;
//uniform vec3 point_light_position;
//uniform vec3 point_light_color;
//uniform vec3 point_light_attenuation;

/*vec3 diffuse(vec3 direction, vec4 albedo_color) {
    return albedo_color.xyz * max(0.0, dot(normal, direction));
}

vec3 specular(vec3 direction, vec4 albedo_color) {
    float power = 1 / (roughness * roughness) - 1;
    vec3 view_direction = normalize(camera_position - position);
    vec3 reflected = 2. * normal * dot(normal, direction) - direction;
    return glossiness * albedo_color.xyz * pow(max(0.0, dot(reflected, view_direction)), power);
}*/

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
    float diffuse_const = max(0.0, dot(normalize(normal), light_direction));
    //vec3 sun = (diffuse(light_direction, albedo_color) + specular(light_direction, albedo_color)) * vec3(1.f, 0.9f, 0.8f);
    //float r = distance(position, point_light_position);
    //vec3 dir = normalize(point_light_position - position);
    //vec3 source = (diffuse(dir, albedo_color) + specular(dir_albedo_color, albedo_color)) * point_light_color / dot(point_light_attenuation, vec3(1, r, r*r));

    out_color = vec4(albedo_color.rgb * (ambient + diffuse_const), albedo_color.a);
}
)";

GLuint create_shader(GLenum type, const char * source)
{
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

template <typename ... Shaders>
GLuint create_program(Shaders ... shaders)
{
    GLuint result = glCreateProgram();
    (glAttachShader(result, shaders), ...);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

glm::vec3 get_or_default_translation(const gltf_model::spline<glm::vec3>& spline, float time) {
    if (spline.values.empty())
        return glm::vec3(0);
    return spline(time);
};

glm::vec3 get_or_default_scale(const gltf_model::spline<glm::vec3>& spline, float time) {
    if (spline.values.empty())
        return glm::vec3(1);
    return spline(time);
};

glm::quat get_or_default_rotation(const gltf_model::spline<glm::quat>& spline, float time) {
    if (spline.values.empty())
        return {1, 0, 0, 0};
    return spline(time);
};


int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 16);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 11",
                                           SDL_WINDOWPOS_CENTERED,
                                           SDL_WINDOWPOS_CENTERED,
                                           800, 600,
                                           SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint albedo_location = glGetUniformLocation(program, "albedo");
    GLuint color_location = glGetUniformLocation(program, "color");
    GLuint use_texture_location = glGetUniformLocation(program, "use_texture");
    GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
    GLuint bones_location = glGetUniformLocation(program, "bones");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint is_rigged_location = glGetUniformLocation(program, "is_rigged");


    const std::string project_root = PROJECT_ROOT;

    const int N_MODELS = 2;
    const std::string model_path[] = {
            project_root + "/models/padoru_santa_saber_alter/scene.gltf",
            project_root + "/models/sparrow_-_quirky_series/scene.gltf"
    };
    gltf_model const input_model[] = {
            load_gltf(model_path[0]),
            load_gltf(model_path[1])
    };

    struct mesh
    {
        GLuint vao;
        gltf_model::accessor indices;
        gltf_model::material material;
    };

    GLuint vbo[N_MODELS];
    glGenBuffers(N_MODELS, vbo);

    std::vector<mesh> meshes[N_MODELS];
    std::map<std::string, GLuint> textures[N_MODELS];

    for (int idx_model = 0; idx_model < N_MODELS; ++idx_model) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo[idx_model]);
        glBufferData(GL_ARRAY_BUFFER, input_model[idx_model].buffer.size(), input_model[idx_model].buffer.data(), GL_STATIC_DRAW);

        auto setup_attribute = [](int index, gltf_model::accessor const & accessor, bool integer = false)
        {
            glEnableVertexAttribArray(index);
            if (integer)
                glVertexAttribIPointer(index, accessor.size, accessor.type, accessor.view.stride, reinterpret_cast<void *>(accessor.view.offset + accessor.offset));
            else
                glVertexAttribPointer(index, accessor.size, accessor.type, GL_FALSE, accessor.view.stride, reinterpret_cast<void *>(accessor.view.offset + accessor.offset));
        };


        glBindBuffer(GL_ARRAY_BUFFER, vbo[idx_model]);
        for (auto const & mesh : input_model[idx_model].meshes)
        {
#ifdef DEBUG
            std::cout << "Joins of current mesh: \nsize: "
                << mesh.joints.size << "\ncount: " << mesh.joints.count << "\ntype: " << mesh.joints.type << '\n';
#endif
            auto & result = meshes[idx_model].emplace_back();
            glGenVertexArrays(1, &result.vao);
            glBindVertexArray(result.vao);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[idx_model]);
            result.indices = mesh.indices;

            setup_attribute(0, mesh.position);
            setup_attribute(1, mesh.normal);
            setup_attribute(2, mesh.texcoord);
            setup_attribute(3, mesh.joints, true);
            setup_attribute(4, mesh.weights);

            result.material = mesh.material;
        }


        for (auto const & mesh : meshes[idx_model])
        {
            if (!mesh.material.texture_path) continue;
            if (textures[idx_model].contains(*mesh.material.texture_path)) continue;

            auto path = std::filesystem::path(model_path[idx_model]).parent_path() / *mesh.material.texture_path;

            int width, height, channels;
            auto data = stbi_load(path.c_str(), &width, &height, &channels, 4);
            assert(data);

            GLuint texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);

            stbi_image_free(data);

            textures[idx_model][*mesh.material.texture_path] = texture;
        }
    }

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f, start_of_shift = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    //float view_angle = glm::pi<float>() / 8.f;
    //float camera_distance = 3.f;
    float animation_interpolation = 0.0f;
    float speed = 5.f;

    float camera_rotation = 0;
    glm::vec3 camera_position{0.f, 1.5f, 3.f};
    float camera_height = 0.25f;

    bool paused = false;
    bool running = true;

    auto ptr_animation = input_model[1].animations.find("Spin");
    if (ptr_animation == input_model[1].animations.end()) {
        throw std::runtime_error("Didn't find Spin animation for bird.");
    }
    auto spin_bird_animation = ptr_animation->second;

    ptr_animation = input_model[1].animations.find("Idle_A");
    if (ptr_animation == input_model[1].animations.end()) {
        throw std::runtime_error("Didn't find Idle_A animation for bird.");
    }
    auto idle_a_bird_animation = ptr_animation->second;

    while (running)
    {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
            {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_WINDOWEVENT: switch (event.window.event)
                    {
                        case SDL_WINDOWEVENT_RESIZED:
                            width = event.window.data1;
                            height = event.window.data2;
                            glViewport(0, 0, width, height);
                            break;
                    }
                    break;
                case SDL_KEYDOWN:
                    button_down[event.key.keysym.sym] = true;
                    if (event.key.keysym.sym == SDLK_SPACE)
                        paused = !paused;
                    if (event.key.keysym.sym == SDLK_LSHIFT)
                        start_of_shift = time;
                    break;
                case SDL_KEYUP:
                    button_down[event.key.keysym.sym] = false;
                    break;
            }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        float camera_move_forward = 0.f;
        float camera_move_sideways = 0.f;

        if (!paused)
            time += dt;
        {
            if (button_down[SDLK_w])
                camera_move_forward -= 3.f * dt;
            if (button_down[SDLK_s])
                camera_move_forward += 3.f * dt;
            if (button_down[SDLK_a])
                camera_move_sideways -= 3.f * dt;
            if (button_down[SDLK_d])
                camera_move_sideways += 3.f * dt;

            if (button_down[SDLK_LEFT])
                camera_rotation -= 3.f * dt;
            if (button_down[SDLK_RIGHT])
                camera_rotation += 3.f * dt;

            if (button_down[SDLK_f])
                camera_position.y -= 4.f * dt;
            if (button_down[SDLK_r])
                camera_position.y += 4.f * dt;

            if (button_down[SDLK_LSHIFT])
                animation_interpolation += speed * dt;
            else
                animation_interpolation -= speed * dt;
            if (animation_interpolation > 1)
                animation_interpolation = 1;
            if (animation_interpolation < 0)
                animation_interpolation = 0;

            camera_position += camera_move_forward * glm::vec3(-std::sin(camera_rotation), 0.f, std::cos(camera_rotation));
            camera_position += camera_move_sideways * glm::vec3(std::cos(camera_rotation), 0.f, std::sin(camera_rotation));
        }


        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        float near = 0.1f;
        float far = 100.f;

        float padoru_turning_angle = time * glm::pi<float>() * 2;

        glm::mat4 model(1.f);

        glm::mat4 view(1.f);
        //view = glm::translate(view, {0.f, 0.f, -camera_distance});
        //view = glm::rotate(view, view_angle, {1.f, 0.f, 0.f});
        view = glm::rotate(view, camera_rotation, {0.f, 1.f, 0.f});
        view = glm::translate(view, -camera_position);

        glm::mat4 projection = glm::perspective(
                glm::pi<float>() / 2.f,
                (1.f * width) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::vec3 light_direction = glm::normalize(glm::vec3(1.f, 2.f, 3.f));

        glUseProgram(program);
        glm::mat4 padoru_view = glm::rotate(view, -glm::pi<float>() / 2, {1.f, 0.f, 0.f});
        padoru_view = glm::rotate(padoru_view, padoru_turning_angle, {0.f, 0.f, 1.f});

        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniform1i(is_rigged_location, 0);
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&padoru_view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
        glUniform3fv(camera_position_location, 1, (float *) (&camera_position));

        auto draw_meshes = [&](bool transparent, int idx_index)
        {
            for (auto const & mesh : meshes[idx_index])
            {
                if (mesh.material.transparent != transparent)
                    continue;

                if (mesh.material.two_sided)
                    glDisable(GL_CULL_FACE);
                else
                    glEnable(GL_CULL_FACE);

                if (transparent)
                    glEnable(GL_BLEND);
                else
                    glDisable(GL_BLEND);

                if (mesh.material.texture_path)
                {
                    glBindTexture(GL_TEXTURE_2D, textures[idx_index][*mesh.material.texture_path]);
                    glUniform1i(use_texture_location, 1);
                }
                else if (mesh.material.color)
                {
                    glUniform1i(use_texture_location, 0);
                    glUniform4fv(color_location, 1, reinterpret_cast<const float *>(&(*mesh.material.color)));
                }
                else
                    continue;

                //glUniform1f(roughness_location, mesh.material.roughnessFactor);
                //glUniform1f(glossiness_location, mesh.material.metallicFactor);

                glBindVertexArray(mesh.vao);
                glDrawElements(GL_TRIANGLES, mesh.indices.count, mesh.indices.type,
                               reinterpret_cast<void *>(mesh.indices.view.offset));
            }
        };

        //glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        draw_meshes(false, 0);
        glDepthMask(GL_FALSE);
        draw_meshes(true, 0);
        glDepthMask(GL_TRUE);

        //glm::mat4 bird_view(1.f);
        //glm::mat4 bird_view = view;
        glm::mat4 bird_view = glm::translate(view, glm::vec3(-3,  0.5, -1));
        //bird_view = glm::rotate(bird_view, -glm::pi<float>() / 2, {1.f, 0.f, 0.f});

        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&bird_view));

        std::vector<glm::mat4x3> bones_matrix(input_model[1].bones.size(), glm::mat4x3(1));
        std::vector<glm::mat4> transforms(idle_a_bird_animation.bones.size());

        for (size_t i = 0; i < spin_bird_animation.bones.size(); ++i) {
            const auto& cur_bone = spin_bird_animation.bones[i];
            auto spin_animation_time = std::fmod(time - start_of_shift, spin_bird_animation.max_time);
            auto spin_translation = get_or_default_translation(cur_bone.translation, spin_animation_time);
            auto spin_rotation = get_or_default_rotation(cur_bone.rotation, spin_animation_time);
            auto spin_scale = get_or_default_scale(cur_bone.scale, spin_animation_time);

            if (i == 0) {
                spin_translation += glm::vec3(0, -0.5, 0);
                spin_rotation = glm::rotate(spin_rotation, -glm::pi<float>() / 2, {1.f, 0.f, 0.f});
            }
            const auto& idle_bone = idle_a_bird_animation.bones[i];
            auto idle_animation_time = std::fmod(time, idle_a_bird_animation.max_time);
            auto idle_translation = get_or_default_translation(idle_bone.translation, idle_animation_time);
            auto idle_rotation = get_or_default_rotation(idle_bone.rotation, idle_animation_time);
            auto idle_scale = get_or_default_scale(idle_bone.scale, idle_animation_time);

            auto translation = glm::lerp(idle_translation, spin_translation, animation_interpolation);
            auto rotation = glm::slerp(idle_rotation, spin_rotation, animation_interpolation);
            auto scale = glm::lerp(idle_scale, spin_scale, animation_interpolation);

            auto glm_translate = glm::translate(glm::mat4(1.f), translation);
            auto glm_rotation = glm::toMat4(rotation);
            auto glm_scale = glm::scale(glm::mat4(1.f), scale);


            transforms[i] = glm_translate * glm_rotation * glm_scale; //glm_translate * glm_rotation * glm_scale
            auto parent = input_model[1].bones[i].parent;
            if (parent != -1) {
                transforms[i] = transforms[parent] * transforms[i];
            }

            bones_matrix[i] = transforms[i];

#ifdef DEBUG
            if (i == 2) {
                std::cout << "Rotation: " << glm::to_string(rotation) << ' '
                          << glm::to_string(idle_rotation) << ' '
                          << glm::to_string(spin_rotation) << '\n';
            }
//            std::cout << "Cur time: " << time << " " << start_of_shift << "\n";
/*            std::cout << "Cur bone id: " << i << " name: " << input_model[1].bones[i].name << '\n';
            std::cout << "Spin animation time: " << spin_animation_time
                      << " of max time: " << spin_bird_animation.max_time << '\n';
            std::cout << "Spin bone " <<
                      " T: " << glm::to_string(spin_translation) << " R: " << glm::to_string(spin_rotation) <<
                      " S: " << glm::to_string(spin_scale) << "\n";
            std::cout << "Idle animation time: " << idle_animation_time
                      << " of max time: " << idle_a_bird_animation.max_time << '\n';
            std::cout << "Idle bone " <<
                      " T: " << glm::to_string(idle_translation) << " R: " << glm::to_string(idle_rotation) <<
                      " S: " << glm::to_string(idle_scale) << "\n";*/
#endif
        }
        for (size_t i = 0; i < input_model[1].bones.size(); ++i) {
            bones_matrix[i] = bones_matrix[i] * input_model[1].bones[i].inverse_bind_matrix;
        }

        glUniformMatrix4x3fv(bones_location, bones_matrix.size(), GL_FALSE, reinterpret_cast<float *>(bones_matrix.data()));

        glUniform1i(is_rigged_location, 1);
        draw_meshes(false, 1);
        glDepthMask(GL_FALSE);
        draw_meshes(true, 1);
        glDepthMask(GL_TRUE);

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
