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
#include "shaders.h"
#include "frustum.hpp"
#include "aabb.hpp"
#include "intersect.hpp"
#include "msdf_loader.h"

const int LEVELS_DETAILS = 6;

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

GLuint load_texture(std::string const & path)
{
    int width, height, channels;
    auto pixels = stbi_load(path.data(), &width, &height, &channels, 4);

    GLuint result;
    glGenTextures(1, &result);
    glBindTexture(GL_TEXTURE_2D, result);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(pixels);

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
}

glm::vec3 get_or_default_scale(const gltf_model::spline<glm::vec3>& spline, float time) {
    if (spline.values.empty())
        return glm::vec3(1);
    return spline(time);
}

glm::quat get_or_default_rotation(const gltf_model::spline<glm::quat>& spline, float time) {
    if (spline.values.empty())
        return {1, 0, 0, 0};
    return spline(time);
}

static glm::vec3 cube_vertices[]
        {
                {0.f, 0.f, 0.f},
                {1.f, 0.f, 0.f},
                {0.f, 1.f, 0.f},
                {1.f, 1.f, 0.f},
                {0.f, 0.f, 1.f},
                {1.f, 0.f, 1.f},
                {0.f, 1.f, 1.f},
                {1.f, 1.f, 1.f},
        };

static std::uint32_t cube_indices[]
        {
                // -Z
                0, 2, 1,
                1, 2, 3,
                // +Z
                4, 5, 6,
                6, 5, 7,
                // -Y
                0, 1, 4,
                4, 1, 5,
                // +Y
                2, 6, 3,
                3, 6, 7,
                // -X
                0, 4, 2,
                2, 4, 6,
                // +X
                1, 3, 5,
                5, 3, 7,
        };


int main() try
{
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
    }

    SDL_Window *window = SDL_CreateWindow("Graphics course practice 11",
                                           SDL_WINDOWPOS_CENTERED,
                                           SDL_WINDOWPOS_CENTERED,
                                           800, 600,
                                           SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    {
        if (!gl_context)
            sdl2_fail("SDL_GL_CreateContext: ");

        if (auto result = glewInit(); result != GLEW_NO_ERROR)
            glew_fail("glewInit: ", result);

        if (!GLEW_VERSION_3_3)
            throw std::runtime_error("OpenGL 3.3 is not supported");
    }

    auto vertex_shader_env = create_shader(GL_VERTEX_SHADER, vertex_shader_env_source);
    auto fragment_shader_env = create_shader(GL_FRAGMENT_SHADER, fragment_shader_env_source);
    auto program_env = create_program(vertex_shader_env, fragment_shader_env);
    GLuint view_location_env = glGetUniformLocation(program_env, "view");
    GLuint projection_location_env = glGetUniformLocation(program_env, "projection");
    GLuint camera_position_location_env = glGetUniformLocation(program_env, "camera_position");
    GLuint environment_texture_location_env = glGetUniformLocation(program_env, "environment_texture");
    GLuint array_env;
    glGenVertexArrays(1, &array_env);

    GLuint vbo_shifts;
    glGenBuffers(1, &vbo_shifts);

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
    GLuint roughness_location = glGetUniformLocation(program, "roughness");
    GLuint is_instance_location = glGetUniformLocation(program, "is_instance");
    GLuint instance_turn_location = glGetUniformLocation(program, "instance_turn");


    const std::string project_root = PROJECT_ROOT;

    GLuint environment_texture = load_texture(project_root + "/textures/environments/environment_map.jpg");

    const int N_MODELS = 3;
    const std::string model_path[] = {
            project_root + "/models/padoru_with_lod/scene.gltf",
            project_root + "/models/sparrow_-_quirky_series/scene.gltf",
            project_root + "/models/disco_ball/scene.gltf"
    };
    gltf_model  input_model[] = { ///REMOVE CONST, maybe it's dangerous //!!!!!!!!!!!!!!!!!!!!!!!!
            load_gltf(model_path[0]),
            load_gltf(model_path[1]),
            load_gltf(model_path[2])
    };
    input_model[2].meshes.pop_back(); ///Only discoball

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
            glEnableVertexAttribArray(5);

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

    SDL_StartTextInput();

    std::map<SDL_Keycode, bool> button_down;

    //float view_angle = glm::pi<float>() / 8.f;
    //float camera_distance = 3.f;
    float animation_interpolation = 0.0f;
    float speed = 5.f;

    float camera_rotation = -1.f;
    glm::vec3 camera_position{25.f, 5.f, 18.f};
    float camera_height = 0.25f;

    bool paused = false;
    bool enter_text = false;
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

    std::vector<glm::vec3> shifts[LEVELS_DETAILS]; ///For instance


    auto vertex_shader_simple = create_shader(GL_VERTEX_SHADER, vertex_shader_source_simple);
    auto fragment_shader_simple = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source_simple);
    auto program_simple = create_program(vertex_shader_simple, fragment_shader_simple);
    glUseProgram(program_simple);
    GLuint view_location_simple = glGetUniformLocation(program_simple, "view");
    GLuint projection_location_simple = glGetUniformLocation(program_simple, "projection");
    GLuint bbox_min_location = glGetUniformLocation(program_simple, "bbox_min");
    GLuint bbox_max_location = glGetUniformLocation(program_simple, "bbox_max");

    GLuint vao_cube, vbo_cube, ebo_cube;
    glGenVertexArrays(1, &vao_cube);
    glBindVertexArray(vao_cube);

    glGenBuffers(1, &vbo_cube);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &ebo_cube);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_cube);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    const glm::vec3 cloud_bbox_min{-30.f, -1.f, -20.f};
    const glm::vec3 cloud_bbox_max{ 30.f,  0.f,  20.f};

    auto msdf_vertex_shader = create_shader(GL_VERTEX_SHADER, msdf_vertex_shader_source);
    auto msdf_fragment_shader = create_shader(GL_FRAGMENT_SHADER, msdf_fragment_shader_source);
    auto msdf_program = create_program(msdf_vertex_shader, msdf_fragment_shader);

    GLuint transform_text_location = glGetUniformLocation(msdf_program, "transform");
    GLuint sdf_scale_text_location = glGetUniformLocation(msdf_program, "sdf_scale");
    struct vertex {
        std::array<float, 2> position;
        std::array<float, 2> texcoord;
    };
    GLuint vao_text;
    glGenVertexArrays(1, &vao_text);
    glBindVertexArray(vao_text);

    GLuint vbo_text;
    glGenBuffers(1, &vbo_text);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), reinterpret_cast<void *>(8));
    const std::string font_path = project_root + "/fonts/font-msdf.json";
    auto const font = load_msdf_font(font_path);
    GLuint texture_text;
    int texture_width_text, texture_height_text;
    {
        int channels;
        auto data = stbi_load(font.texture_path.c_str(), &texture_width_text, &texture_height_text, &channels, 4);
        assert(data);

        glGenTextures(1, &texture_text);
        glBindTexture(GL_TEXTURE_2D, texture_text);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texture_width_text, texture_height_text, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(data);
    }
    int number_of_trim_texts = 0;
    std::string text = "Disco time ~~~";
    bool text_changed = true;

    /*GLsizei shadow_map_resolution = 1024;

    GLuint shadow_map;
    glGenTextures(1, &shadow_map);
    glBindTexture(GL_TEXTURE_2D, shadow_map);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_map_resolution, shadow_map_resolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    GLuint shadow_fbo;
    glGenFramebuffers(1, &shadow_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, shadow_fbo);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadow_map, 0);
    if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Incomplete framebuffer!");
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);*/

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
                    if (event.key.keysym.sym == SDLK_t && !enter_text)
                        enter_text = !enter_text;
                    if (event.key.keysym.sym == SDLK_ESCAPE && enter_text)
                        enter_text = !enter_text;
                    if (event.key.keysym.sym == SDLK_BACKSPACE && !text.empty() && enter_text)
                    {
                        text.pop_back();
                        text_changed = true;
                    }
                    break;
                case SDL_TEXTINPUT:
                    if (enter_text) {
                        text.append(event.text.text);
                        text_changed = true;
                    }
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
        if (!enter_text)
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

        float near = 0.1f;
        float far = 100.f;

        glm::mat4 model(1.f);

        glm::mat4 view(1.f);
        view = glm::rotate(view, camera_rotation, {0.f, 1.f, 0.f});
        view = glm::translate(view, -camera_position);

        glm::mat4 projection = glm::perspective(
                glm::pi<float>() / 2.f,
                (1.f * width) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::vec3 light_direction = glm::normalize(glm::vec3(1.f, 2.f, 3.f));

        //glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //glViewport(0, 0, width, height);

        glUseProgram(program_env);
        glUniformMatrix4fv(view_location_env, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location_env, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(camera_position_location_env, 1, reinterpret_cast<float *>(&camera_position));
        glUniform1i(environment_texture_location_env, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, environment_texture);
        glBindVertexArray(array_env);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        glUseProgram(program);
        float padoru_turning_angle = time * glm::pi<float>() * 2;
        glm::mat4 padoru_view = view;
        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));
        glUniform1i(is_rigged_location, 0);
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&padoru_view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(light_direction_location, 1, reinterpret_cast<float *>(&light_direction));
        glUniform3fv(camera_position_location, 1, (float *) (&camera_position));

        auto draw_meshes = [&](bool transparent, int idx_index,
                                                  glm::mat4 turn_view, bool is_instance = false,
                                                  int dx_minus = 0, int dx_plus = 0,
                                                  int dz_minus = 0, int dz_plus = 0)
        {
            glUniform1i(is_instance_location, is_instance);
            if (is_instance) {
                glm::vec3 min = input_model[idx_index].meshes[0].min, max = input_model[idx_index].meshes[0].max;
                aabb(min, max);
                for (auto & shift : shifts)
                    shift.clear();
                const float LOD_CONST_LENGTH[LEVELS_DETAILS - 1] = {10, 20, 30, 40, 50};
                for (int x = dx_minus; x <= dx_plus; ++x) {
                    for (int z = dz_minus; z <= dz_plus; ++z) {
                        auto center = glm::vec3(x * 5, 0, z * 5);
                        glm::translate(glm::mat4(1), center);
                        auto cur_aabb = aabb(min + center, max + center);
                        glm::mat4 instance_view = glm::translate(view, center);
                        instance_view = instance_view * turn_view;
                        instance_view = glm::translate(instance_view, -center);
                        auto cur_frustum = frustum(projection * instance_view);
                        if (intersect(cur_aabb, cur_frustum)){
                            float len = glm::length(center - camera_position);
                            int index = 0;
                            while (index < LEVELS_DETAILS - 1 && index + 1 < meshes[idx_index].size() && len > LOD_CONST_LENGTH[index]) {
                                ++index;
                            }
                            shifts[index].emplace_back(center);
                        }
                    }
                }
            }

            for (size_t i = 0; i < meshes[idx_index].size(); ++i)
            {
                auto const &mesh = meshes[idx_index][i];
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

                glUniform1f(roughness_location, mesh.material.roughnessFactor);

                if (!is_instance) {
                    glBindVertexArray(mesh.vao);
                    glDrawElements(GL_TRIANGLES, mesh.indices.count, mesh.indices.type,
                                   reinterpret_cast<void *>(mesh.indices.view.offset));
                } else {
                    auto &shift = shifts[i];
                    glBindVertexArray(mesh.vao);
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_shifts);
                    glBufferData(GL_ARRAY_BUFFER, shift.size() * sizeof(shift[0]), shift.data(), GL_STATIC_DRAW);
                    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void *>(0));
                    glVertexAttribDivisor(5, 1);
                    glUniformMatrix4fv(instance_turn_location, 1, GL_FALSE, reinterpret_cast<float *>(&turn_view));
                    glDrawElementsInstanced(GL_TRIANGLES, mesh.indices.count, mesh.indices.type,
                                            reinterpret_cast<void *>(mesh.indices.view.offset), shift.size());
                }
            }

        };

        glm::mat4 turn_view_padoru = glm::mat4(1);
        turn_view_padoru = glm::rotate(turn_view_padoru, -glm::pi<float>() / 2, {1.f, 0.f, 0.f});
        turn_view_padoru = glm::rotate(turn_view_padoru, padoru_turning_angle, {0.f, 0.f, 1.f});

        draw_meshes(false, 0, turn_view_padoru, true, -5, 5, -5, 5);
        glDepthMask(GL_FALSE);
        draw_meshes(true, 0, turn_view_padoru, true, -5, 5, -5, 5);
        glDepthMask(GL_TRUE);
        //glm::mat4 bird_view(1.f);
        //glm::mat4 bird_view = view;
        glm::mat4 bird_view = glm::translate(view, glm::vec3(-5,  2, -1));
        bird_view = glm::rotate(bird_view, -camera_rotation, {0.f, 1.f, 0.f});
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
        }
        for (size_t i = 0; i < input_model[1].bones.size(); ++i) {
            bones_matrix[i] = bones_matrix[i] * input_model[1].bones[i].inverse_bind_matrix;
        }

        glUniformMatrix4x3fv(bones_location, bones_matrix.size(), GL_FALSE, reinterpret_cast<float *>(bones_matrix.data()));

        glUniform1i(is_rigged_location, 1);
        draw_meshes(false, 1, bird_view);
        glDepthMask(GL_FALSE);
        draw_meshes(true, 1, bird_view);
        glDepthMask(GL_TRUE);

        glm::mat4 disco_view = glm::translate(view, glm::vec3(0.,  15., 0.));
        disco_view = glm::rotate(disco_view, -glm::pi<float>() / 2, {1.f, 0.f, 0.f});
        disco_view = glm::rotate(disco_view, padoru_turning_angle, {0.f, 0.f, 1.f});
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&disco_view));

        glUniform1i(is_rigged_location, 0);
        draw_meshes(false, 2, disco_view);
        glDepthMask(GL_FALSE);
        draw_meshes(true, 2, disco_view);
        glDepthMask(GL_TRUE);

        glUseProgram(program_simple);
        glUniformMatrix4fv(view_location_simple, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location_simple, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(bbox_min_location, 1, reinterpret_cast<const float *>(&cloud_bbox_min));
        glUniform3fv(bbox_max_location, 1, reinterpret_cast<const float *>(&cloud_bbox_max));
        glBindVertexArray(vao_cube);
        glDrawElements(GL_TRIANGLES, std::size(cube_indices), GL_UNSIGNED_INT, nullptr);
        glUseProgram(msdf_program);
        glDisable(GL_CULL_FACE);
        glm::mat4 transform_text = glm::mat4(1);
        transform_text = glm::translate(transform_text, glm::vec3(-1, +1, 0));
        transform_text = glm::scale(transform_text, glm::vec3(2. / width, -2. / height, 1.));
        glUniformMatrix4fv(transform_text_location, 1, GL_FALSE, reinterpret_cast<float *>(&transform_text));
        glUniform1f(sdf_scale_text_location, font.sdf_scale);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_text);

        if (text_changed) {
            number_of_trim_texts = 0;
            std::vector<vertex> vertexes;
            glm::vec2 pen(0.0);
            for (char ch : text) {
                if (ch == '\0')
                    break;
                const msdf_font::glyph glyph = font.glyphs.at(ch);

                auto make_glyph_vertex = [&](float dx, float dy) -> vertex {
                    return {{glyph.xoffset + dx + pen.x, glyph.yoffset + dy + pen.y},
                            {(glyph.x + dx) / texture_width_text, (glyph.y + dy) / texture_height_text}};
                };

                vertexes.push_back(make_glyph_vertex(0, 0));
                vertexes.push_back(make_glyph_vertex(glyph.width, 0));
                vertexes.push_back(make_glyph_vertex(0, glyph.height));
                vertexes.push_back(make_glyph_vertex(glyph.width, glyph.height));
                vertexes.push_back(make_glyph_vertex(0, glyph.height));
                vertexes.push_back(make_glyph_vertex(glyph.width, 0));
                number_of_trim_texts += 6;
                pen.x += glyph.advance;
            }

            glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
            glBufferData(GL_ARRAY_BUFFER, vertexes.size() * sizeof(vertex), vertexes.data(), GL_STATIC_DRAW);
            text_changed = false;
        }

        glBindVertexArray(vao_text);
        glDrawArrays(GL_TRIANGLES, 0, number_of_trim_texts);

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
