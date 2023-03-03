cmake_minimum_required(VERSION 3.0)
project(project)

cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0074 NEW)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/FBXImport)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

set(CMAKE_CXX_STANDARD 20)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(CMakePrintHelpers)

cmake_print_variables(SDL2_ROOT GLEW_ROOT)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(SDL2 REQUIRED)
cmake_print_variables(SDL2_INCLUDE_DIRS GLEW_INCLUDE_DIRS OPENGL_INCLUDE_DIRS)

if(APPLE)
    # brew version of glew doesn't provide GLEW_* variables
    get_target_property(GLEW_INCLUDE_DIRS GLEW::GLEW INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(GLEW_LIBRARIES GLEW::GLEW INTERFACE_LINK_LIBRARIES)
    get_target_property(GLEW_LIBRARY GLEW::GLEW LOCATION)
    list(APPEND GLEW_LIBRARIES "${GLEW_LIBRARY}")
endif()

set(TARGET_NAME "${PROJECT_NAME}")

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")


add_executable(${TARGET_NAME} main.cpp
        gltf_loader.hpp
        gltf_loader.cpp
        stb_image.h
        stb_image.c
        intersect.hpp
        aabb.hpp
        aabb.cpp
        frustum.hpp
        frustum.cpp
        )
target_include_directories(${TARGET_NAME} PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/rapidjson/include"
        "${SDL2_INCLUDE_DIRS}"
        "${GLEW_INCLUDE_DIRS}"
        "${OPENGL_INCLUDE_DIRS}"
        #${CMAKE_CURRENT_SOURCE_DIR}/libs/FBXImport/importer/include
        )
target_link_libraries(${TARGET_NAME} PUBLIC
        "${GLEW_LIBRARIES}"
        "${SDL2_LIBRARIES}"
        "${OPENGL_LIBRARIES}"
        #FBXImport
        Threads::Threads
        )
target_compile_definitions(${TARGET_NAME} PUBLIC
        -DPROJECT_ROOT="${PROJECT_ROOT}"
        -DGLM_FORCE_SWIZZLE
        -DGLM_ENABLE_EXPERIMENTAL
        )

