#include "gltf_loader.hpp"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <fstream>
#include <stdexcept>
#include <GL/glew.h>


static unsigned int attribute_type_to_size(std::string const & type)
{
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    return 0;
}

gltf_model load_gltf(std::filesystem::path const & path)
{
    rapidjson::Document document;

    {
        std::ifstream input(path, std::ios::binary);
        rapidjson::IStreamWrapper stream(input);
        document.ParseStream(stream);
    }

    gltf_model result;

    {
        auto buffers = document["buffers"].GetArray();
        assert(buffers.Size() == 1);

        auto const buffer_uri = buffers[0]["uri"].GetString();

        auto const buffer_path = path.parent_path() / buffer_uri;

        result.buffer.resize(std::filesystem::file_size(buffer_path));
        std::ifstream buffer(buffer_path, std::ios::binary);
        buffer.read(result.buffer.data(), result.buffer.size());
    }

    auto parse_buffer_view = [&](int index) -> gltf_model::buffer_view
    {
        auto view = document["bufferViews"].GetArray()[index].GetObject();
        unsigned int offset = 0, stride = 0;
        if (view.HasMember("byteOffset")) {
            offset = view["byteOffset"].GetUint();
        }
        if (view.HasMember("byteStride")) {
            stride = view["byteStride"].GetUint();
        }
        return {offset, view["byteLength"].GetUint(), stride};
    };

    auto parse_accessor = [&](int index) -> gltf_model::accessor
    {
        unsigned int offset = 0;
        auto accessor = document["accessors"].GetArray()[index].GetObject();
        if (accessor.HasMember("byteOffset")) {
            offset = accessor["byteOffset"].GetUint();
        }
        return {
                parse_buffer_view(accessor["bufferView"].GetInt()),
                accessor["componentType"].GetUint(),
                attribute_type_to_size(accessor["type"].GetString()),
                accessor["count"].GetUint(),
                offset,
        };
    };

    auto parse_texture = [&](int index) -> std::string
    {
        auto const source_index = document["textures"].GetArray()[index]["source"].GetInt();
        return document["images"].GetArray()[source_index]["uri"].GetString();
    };

    auto parse_color = [&](auto const & array)
    {
        return glm::vec4{
                array[0].GetFloat(),
                array[1].GetFloat(),
                array[2].GetFloat(),
                array[3].GetFloat(),
        };
    };

    auto parse_vector = [&](auto const & array)
    {
        return glm::vec3{
                array[0].GetFloat(),
                array[1].GetFloat(),
                array[2].GetFloat(),
        };
    };

    auto parse_bounds = [&](int index)
    {
        auto accessor = document["accessors"].GetArray()[index].GetObject();
        return std::make_pair(
                parse_vector(accessor["min"]),
                parse_vector(accessor["max"])
        );
    };

    for (auto const & mesh : document["meshes"].GetArray()) {
        auto &result_mesh = result.meshes.emplace_back();
        result_mesh.name = mesh["name"].GetString();

        auto primitives = mesh["primitives"].GetArray();
        assert(primitives.Size() == 1);

        auto const &attributes = primitives[0]["attributes"];

        result_mesh.indices = parse_accessor(primitives[0]["indices"].GetInt());
        result_mesh.position = parse_accessor(attributes["POSITION"].GetInt());
        result_mesh.normal = parse_accessor(attributes["NORMAL"].GetInt());
        result_mesh.texcoord = parse_accessor(attributes["TEXCOORD_0"].GetInt());

        result_mesh.is_rigged = false;
        if (attributes.HasMember("JOINTS_0")) {
            result_mesh.is_rigged = true;
            result_mesh.joints = parse_accessor(attributes["JOINTS_0"].GetInt());
            result_mesh.weights = parse_accessor(attributes["WEIGHTS_0"].GetInt());
        }

        std::tie(result_mesh.min, result_mesh.max) = parse_bounds(attributes["POSITION"].GetInt());

        auto const &material = document["materials"].GetArray()[primitives[0]["material"].GetInt()];

        result_mesh.material.two_sided = material.HasMember("doubleSided") && material["doubleSided"].GetBool();
        result_mesh.material.transparent =
                material.HasMember("alphaMode") && (material["alphaMode"].GetString() == std::string("BLEND"));

        auto const &pbr = material["pbrMetallicRoughness"];
        if (pbr.HasMember("baseColorTexture"))
            result_mesh.material.texture_path = parse_texture(pbr["baseColorTexture"]["index"].GetInt());
        else if (pbr.HasMember("baseColorFactor"))
            result_mesh.material.color = parse_color(pbr["baseColorFactor"].GetArray());

        //TODO? for metallicRoughness map
        if (pbr.HasMember("metallicFactor")) {
            result_mesh.material.metallicFactor = pbr["metallicFactor"].GetFloat();
        }
        if (pbr.HasMember("roughnessFactor")) {
            result_mesh.material.roughnessFactor = pbr["roughnessFactor"].GetFloat();
        }
    }

    if (document.HasMember("skins")) {
        auto skins = document["skins"].GetArray();
        assert(skins.Size() == 1);
        {
            auto fill_buffer = [&](auto &vector, gltf_model::accessor const &accessor) {
                assert(accessor.type == 0x1406); // GL_FLOAT
                assert(accessor.view.stride == 0 || sizeof(vector[0]) == accessor.view.stride);
                using value_type = std::decay_t<decltype(vector[0])>;
                assert(accessor.size == 0 || sizeof(value_type) == sizeof(GLfloat) * accessor.size);
                auto begin = reinterpret_cast<value_type const *>(result.buffer.data() + accessor.view.offset + accessor.offset);
                vector.assign(begin, begin + accessor.count);
            };

            auto fix_rotations = [](std::vector<glm::quat> &rotations) {
                for (auto &r: rotations) {
                    r = glm::quat(r.z, r.w, r.x, r.y);
                }
            };

            auto joints = skins[0]["joints"].GetArray();

            std::vector<glm::mat4> inverse_bind_matrices(joints.Size());

            fill_buffer(inverse_bind_matrices, parse_accessor(skins[0]["inverseBindMatrices"].GetInt()));

            result.bones.resize(joints.Size());

            std::unordered_map<int, int> bone_node_to_index;
            for (int i = 0; i < joints.Size(); ++i) {
                int const node_id = joints[i].GetInt();
                bone_node_to_index[node_id] = i;
                result.bones[i].name = document["nodes"].GetArray()[node_id]["name"].GetString();
                result.bones[i].inverse_bind_matrix = inverse_bind_matrices[i];
            }

            auto nodes = document["nodes"].GetArray();

            for (int i = 0; i < nodes.Size(); ++i) {
                if (!bone_node_to_index.contains(i)) continue;

                auto const &node = nodes[i];

                if (!node.HasMember("children")) continue;

                for (auto const &child: node["children"].GetArray()) {
                    int child_id = child.GetInt();
                    if (bone_node_to_index.contains(child_id))
                        result.bones[bone_node_to_index.at(child_id)].parent = bone_node_to_index.at(i);
                }
            }

            for (int i = 0; i < result.bones.size(); ++i)
                assert(result.bones[i].parent == -1 || result.bones[i].parent < i);

            for (auto const &animation: document["animations"].GetArray()) {
                std::string name = animation["name"].GetString();

                auto samplers = animation["samplers"].GetArray();

                gltf_model::animation result_animation;
                result_animation.bones.resize(result.bones.size());

                for (auto const &channel: animation["channels"].GetArray()) {
                    int node_id = channel["target"]["node"].GetInt();
                    if (!bone_node_to_index.contains(node_id)) continue;

                    auto &bone = result_animation.bones[bone_node_to_index.at(node_id)];

                    std::string path = channel["target"]["path"].GetString();

                    auto const &sampler = samplers[channel["sampler"].GetInt()];

                    auto input = parse_accessor(sampler["input"].GetInt());
                    auto output = parse_accessor(sampler["output"].GetInt());

                    if (path == "translation") {
                        fill_buffer(bone.translation.timestamps, input);
                        fill_buffer(bone.translation.values, output);
                    } else if (path == "rotation") {
                        fill_buffer(bone.rotation.timestamps, input);
                        fill_buffer(bone.rotation.values, output);
                        fix_rotations(bone.rotation.values);
                    } else if (path == "scale") {
                        fill_buffer(bone.scale.timestamps, input);
                        fill_buffer(bone.scale.values, output);
                    }
                }

                auto update_max_time = [&](std::vector<float> const &timestamps) {
                    for (float t: timestamps)
                        result_animation.max_time = std::max(result_animation.max_time, t);
                };

                for (auto const &bone: result_animation.bones) {
                    update_max_time(bone.translation.timestamps);
                    update_max_time(bone.rotation.timestamps);
                    update_max_time(bone.scale.timestamps);
                }

                result.animations[std::move(name)] = std::move(result_animation);
            }
        }
    }

    return result;
}
