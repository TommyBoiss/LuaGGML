cmake_minimum_required(VERSION 3.14)

project(lua_ggml LANGUAGES C CXX)

option(LUA_GGML_AUTO_BINDINGS "Generate Lua bindings from ggml headers using clang" ON)

# Ensure static and shared libs are built with PIC so they can be linked into Lua modules.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep the embedded ggml build lean for this module.
set(GGML_BUILD_TESTS OFF CACHE BOOL "Build ggml tests" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Build ggml examples" FORCE)

add_subdirectory(ggml)

find_package(Lua REQUIRED)

set(LUA_GGML_SOURCES lua_ggml.c)

if (LUA_GGML_AUTO_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    find_program(CLANG_EXECUTABLE NAMES clang REQUIRED)

    set(LUA_GGML_BINDINGS_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/ggml.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/ggml-alloc.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/ggml-backend.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/ggml-cpu.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/gguf.h
    )

    set(LUA_GGML_OPTIONAL_BINDING_TARGETS
        ggml-blas
        ggml-cann
        ggml-cuda
        ggml-hexagon
        ggml-hip
        ggml-metal
        ggml-musa
        ggml-opencl
        ggml-rpc
        ggml-sycl
        ggml-virtgpu
        ggml-vulkan
        ggml-webgpu
        ggml-zdnn
        ggml-zendnn
    )

    foreach(optional_target IN LISTS LUA_GGML_OPTIONAL_BINDING_TARGETS)
        if (TARGET ${optional_target})
            set(optional_header ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/${optional_target}.h)
            if (EXISTS ${optional_header})
                list(APPEND LUA_GGML_BINDINGS_HEADERS ${optional_header})
            endif()
        endif()
    endforeach()

    set(LUA_GGML_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
    set(LUA_GGML_GENERATED_C ${LUA_GGML_GENERATED_DIR}/lua_ggml_bindings.c)
    set(LUA_GGML_GENERATED_H ${LUA_GGML_GENERATED_DIR}/lua_ggml_bindings.h)

    set(LUA_GGML_GENERATOR_ARGS
        --header ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/ggml.h
        --header ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/ggml-alloc.h
        --header ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/ggml-backend.h
        --header ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/ggml-cpu.h
        --header ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/gguf.h
        --prefix ggml_
        --prefix gguf_
        --strip-prefix ggml_
        --out-c ${LUA_GGML_GENERATED_C}
        --out-h ${LUA_GGML_GENERATED_H}
        --clang-bin ${CLANG_EXECUTABLE}
        --clang-arg=-I${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
        --nm-library $<TARGET_FILE:ggml-base>
        --nm-library $<TARGET_FILE:ggml>
    )

    if (TARGET ggml-cpu)
        list(APPEND LUA_GGML_GENERATOR_ARGS --nm-library $<TARGET_FILE:ggml-cpu>)
    endif()

    foreach(optional_target IN LISTS LUA_GGML_OPTIONAL_BINDING_TARGETS)
        if (TARGET ${optional_target})
            set(optional_header ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/${optional_target}.h)
            if (EXISTS ${optional_header})
                list(APPEND LUA_GGML_GENERATOR_ARGS --header ${optional_header})
            endif()
            list(APPEND LUA_GGML_GENERATOR_ARGS --nm-library $<TARGET_FILE:${optional_target}>)
        endif()
    endforeach()

    add_custom_command(
        OUTPUT ${LUA_GGML_GENERATED_C} ${LUA_GGML_GENERATED_H}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${LUA_GGML_GENERATED_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_lua_bindings.py
                ${LUA_GGML_GENERATOR_ARGS}
        DEPENDS
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_lua_bindings.py
                ${LUA_GGML_BINDINGS_HEADERS}
        COMMENT "Generating Lua ggml bindings"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )

    add_custom_target(lua_ggml_generate_bindings
        DEPENDS ${LUA_GGML_GENERATED_C} ${LUA_GGML_GENERATED_H}
    )

    add_dependencies(lua_ggml_generate_bindings ggml)
    add_dependencies(lua_ggml_generate_bindings ggml-base)
    if (TARGET ggml-cpu)
        add_dependencies(lua_ggml_generate_bindings ggml-cpu)
    endif()

    list(APPEND LUA_GGML_SOURCES ${LUA_GGML_GENERATED_C})
endif()

add_library(lua_ggml MODULE ${LUA_GGML_SOURCES})
target_compile_features(lua_ggml PRIVATE c_std_11)
target_link_libraries(lua_ggml PRIVATE ggml::ggml)

if (TARGET ggml-cpu)
    target_link_libraries(lua_ggml PRIVATE ggml-cpu)
endif()

if (LUA_GGML_AUTO_BINDINGS)
    add_dependencies(lua_ggml lua_ggml_generate_bindings)
    target_include_directories(lua_ggml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LUA_GGML_GENERATED_DIR})
    target_compile_definitions(lua_ggml PRIVATE LUA_GGML_HAS_GENERATED_BINDINGS=1)
else()
    target_compile_definitions(lua_ggml PRIVATE LUA_GGML_HAS_GENERATED_BINDINGS=0)
endif()

foreach(optional_target IN LISTS LUA_GGML_OPTIONAL_BINDING_TARGETS)
    if (TARGET ${optional_target})
        target_link_libraries(lua_ggml PRIVATE ${optional_target})
    endif()
endforeach()

if (TARGET Lua::Lua)
    target_link_libraries(lua_ggml PRIVATE Lua::Lua)
else()
    target_include_directories(lua_ggml PRIVATE ${LUA_INCLUDE_DIR})
    target_link_libraries(lua_ggml PRIVATE ${LUA_LIBRARIES})
endif()

# Lua's require("ggml") expects ggml.so on Unix-like systems.
set_target_properties(lua_ggml PROPERTIES
    PREFIX ""
    OUTPUT_NAME "ggml"
)

if (UNIX AND NOT APPLE)
    # Let ggml.so find libggml.so directly from the build tree.
    set_target_properties(lua_ggml PROPERTIES
        BUILD_RPATH "$ORIGIN;$ORIGIN/ggml/src"
        INSTALL_RPATH "$ORIGIN"
    )
endif()
