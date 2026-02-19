cmake_minimum_required(VERSION 3.18)

project(tomotopy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SKIP_RPATH ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version")

# Options
option(BUILD_MULTI_ARCH "Build multiple architecture variants with dispatcher" OFF)
option(NO_DEBUG_INFO "Strip debug info for release builds" OFF)

set(TARGET_ARCH "native" CACHE STRING "Target architecture (none, native, sse2, avx, avx2, avx512, arm64)")
set(TOMOTOPY_ISA "native" CACHE STRING "ISA string for module")
set(TOMOTOPY_LANG "en" CACHE STRING "Documentation language (en, kr)")

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

# Source files
file(GLOB HANDLER_SOURCES
    "src/python/handler/*.cpp"
    "src/Utils/*.cpp"
    "src/TopicModel/*.cpp"
    "src/Labeling/*.cpp"
)

file(GLOB DISPATCHER_SOURCES
    "src/python/dispatcher/*.cpp"
)

# Function to get compiler flags for an architecture
function(get_arch_flags ARCH OUT_FLAGS OUT_DEFS)
    set(FLAGS "")
    set(DEFS "")

    if(MSVC)
        if(ARCH STREQUAL "none")
            # No additional flags
        elseif(ARCH STREQUAL "sse2")
            set(FLAGS "/arch:SSE2")
            list(APPEND DEFS "__SSE2__")
        elseif(ARCH STREQUAL "avx")
            set(FLAGS "/arch:AVX")
            list(APPEND DEFS "__SSE2__")
        elseif(ARCH STREQUAL "avx2")
            set(FLAGS "/arch:AVX2")
            list(APPEND DEFS "__SSE2__")
        elseif(ARCH STREQUAL "avx512")
            set(FLAGS "/arch:AVX512")
            list(APPEND DEFS "__SSE2__" "__AVX__" "__AVX2__" "__AVX512F__")
        endif()
    else()
        if(ARCH STREQUAL "native")
            set(FLAGS "-march=native")
        elseif(ARCH STREQUAL "sse2")
            set(FLAGS "-msse2")
        elseif(ARCH STREQUAL "avx")
            set(FLAGS "-mavx")
        elseif(ARCH STREQUAL "avx2")
            set(FLAGS "-mavx2 -mfma")
        elseif(ARCH STREQUAL "avx512")
            set(FLAGS "-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma")
        elseif(ARCH STREQUAL "arm64")
            # No additional flags for arm64
        endif()

        if(APPLE)
            set(FLAGS "${FLAGS} -undefined dynamic_lookup")
        endif()
    endif()

    set(${OUT_FLAGS} "${FLAGS}" PARENT_SCOPE)
    set(${OUT_DEFS} "${DEFS}" PARENT_SCOPE)
endfunction()

# Function to create a handler module
function(add_handler_module TARGET_NAME ARCH ISA_NAME)
    get_arch_flags(${ARCH} ARCH_FLAGS ARCH_DEFS)

    add_library(${TARGET_NAME} MODULE ${HANDLER_SOURCES})

    target_compile_definitions(${TARGET_NAME} PRIVATE
        MODULE_NAME=PyInit_${TARGET_NAME}
        TOMOTOPY_ISA=${ISA_NAME}
        ${ARCH_DEFS}
    )

    if(TOMOTOPY_LANG STREQUAL "kr")
        target_compile_definitions(${TARGET_NAME} PRIVATE DOC_KO=1)
    endif()

    if(ARCH_FLAGS)
        target_compile_options(${TARGET_NAME} PRIVATE ${ARCH_FLAGS})
    endif()

    if(NO_DEBUG_INFO)
        if(MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE /Zi-)
        else()
            target_compile_options(${TARGET_NAME} PRIVATE -g0)
        endif()
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        PREFIX ""
        OUTPUT_NAME ${TARGET_NAME}
    )

    if(WIN32)
        set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".pyd")
    endif()

    target_link_libraries(${TARGET_NAME} PRIVATE Python3::Module)

    install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION .)
endfunction()

# Function to create dispatcher module
function(add_dispatcher_module)
    add_library(_tomotopy MODULE ${DISPATCHER_SOURCES})

    set_target_properties(_tomotopy PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_tomotopy"
    )

    if(WIN32)
        set_target_properties(_tomotopy PROPERTIES SUFFIX ".pyd")
    endif()

    target_link_libraries(_tomotopy PRIVATE Python3::Module)

    install(TARGETS _tomotopy LIBRARY DESTINATION .)
endfunction()

# Build logic
if(BUILD_MULTI_ARCH)
    message(STATUS "Building multi-architecture modules with dispatcher")

    # Build dispatcher
    add_dispatcher_module()

    # Determine which architectures to build
    if(APPLE)
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            add_handler_module("_tomotopy_none" "arm64" "arm64")
        else()
            add_handler_module("_tomotopy_sse2" "sse2" "sse2")
            add_handler_module("_tomotopy_avx" "avx" "avx")
            add_handler_module("_tomotopy_avx2" "avx2" "avx2")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        add_handler_module("_tomotopy_none" "arm64" "arm64")
    else()
        # x86_64 Linux/Windows: build all variants
        add_handler_module("_tomotopy_sse2" "sse2" "sse2")
        add_handler_module("_tomotopy_avx" "avx" "avx")
        add_handler_module("_tomotopy_avx2" "avx2" "avx2")
        add_handler_module("_tomotopy_avx512" "avx512" "avx512")

        if(WIN32)
            # Windows also needs 'none' for 32-bit compatibility
            add_handler_module("_tomotopy_none" "none" "none")
        endif()
    endif()
else()
    message(STATUS "Building single module for architecture: ${TARGET_ARCH}")
    add_handler_module("_tomotopy" "${TARGET_ARCH}" "${TOMOTOPY_ISA}")
endif()
