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
option(USE_STABLE_ABI "Build with Python Stable ABI (abi3)" OFF)

set(TARGET_ARCH "native" CACHE STRING "Target architecture (none, native, sse2, avx, avx2, avx512, arm64)")
set(TOMOTOPY_ISA "native" CACHE STRING "ISA string for module")
set(TOMOTOPY_LANG "en" CACHE STRING "Documentation language (en, kr)")

# Stable ABI setup
if(USE_STABLE_ABI)
    add_compile_definitions(Py_LIMITED_API=0x03090000)
endif()

# Fetch dependencies
include(FetchContent)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_Declare(
    eigenrand
    GIT_REPOSITORY https://github.com/bab2min/EigenRand.git
    GIT_TAG v0.5.1
)

# Populate without add_subdirectory() to avoid installing their targets into our wheel
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()

FetchContent_GetProperties(eigenrand)
if(NOT eigenrand_POPULATED)
    FetchContent_Populate(eigenrand)
endif()

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Include directories
include_directories(
    ${eigen_SOURCE_DIR}
    ${eigenrand_SOURCE_DIR}
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
        # On MSVC x64, SSE2 is always available (baseline)
        # MSVC doesn't auto-define __SSE2__, so we must define it manually
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            list(APPEND DEFS "__SSE2__")
        endif()

        if(ARCH STREQUAL "none")
            # x86 32-bit: no SIMD
            # x64: SSE2 defined above
        elseif(ARCH STREQUAL "sse2")
            set(FLAGS "/arch:SSE2")
            if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                list(APPEND DEFS "__SSE2__")
            endif()
        elseif(ARCH STREQUAL "avx")
            set(FLAGS "/arch:AVX")
            list(APPEND DEFS "__AVX__")
        elseif(ARCH STREQUAL "avx2")
            set(FLAGS "/arch:AVX2")
            list(APPEND DEFS "__AVX__" "__AVX2__")
        elseif(ARCH STREQUAL "avx512")
            set(FLAGS "/arch:AVX512")
            list(APPEND DEFS "__AVX__" "__AVX2__" "__AVX512F__")
        endif()
    else()
        if(ARCH STREQUAL "native")
            set(FLAGS "-march=native")
        elseif(ARCH STREQUAL "sse2")
            set(FLAGS "-msse2")
        elseif(ARCH STREQUAL "avx")
            set(FLAGS "-mavx")
        elseif(ARCH STREQUAL "avx2")
            set(FLAGS "-mavx2" "-mfma")
        elseif(ARCH STREQUAL "avx512")
            set(FLAGS "-mavx512f" "-mavx512dq" "-mavx512bw" "-mavx512vl" "-mfma")
        elseif(ARCH STREQUAL "arm64")
            # No additional flags for arm64
        endif()

        if(APPLE)
            list(APPEND FLAGS "-undefined" "dynamic_lookup")
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

    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /bigobj)
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
    elseif(USE_STABLE_ABI)
        set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".abi3.so")
    endif()

    if(USE_STABLE_ABI AND WIN32)
        # Link against python3.lib (stable ABI import library) instead of python3X.lib
        get_filename_component(_python_lib_dir "${Python3_LIBRARY_DIRS}" ABSOLUTE)
        find_file(PYTHON3_STABLE_LIB "python3.lib" PATHS "${_python_lib_dir}" NO_DEFAULT_PATH)
        if(NOT PYTHON3_STABLE_LIB)
            # Fallback: search in the Python prefix
            find_file(PYTHON3_STABLE_LIB "python3.lib"
                PATHS "${Python3_RUNTIME_LIBRARY_DIRS}" "${Python3_LIBRARY_DIRS}"
                PATH_SUFFIXES "libs" "lib"
            )
        endif()
        if(PYTHON3_STABLE_LIB)
            target_link_libraries(${TARGET_NAME} PRIVATE "${PYTHON3_STABLE_LIB}")
            target_include_directories(${TARGET_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
        else()
            message(WARNING "python3.lib not found, falling back to Python3::Module")
            target_link_libraries(${TARGET_NAME} PRIVATE Python3::Module)
        endif()
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE Python3::Module)
    endif()

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
    elseif(USE_STABLE_ABI)
        set_target_properties(_tomotopy PROPERTIES SUFFIX ".abi3.so")
    endif()

    if(USE_STABLE_ABI AND WIN32)
        if(PYTHON3_STABLE_LIB)
            target_link_libraries(_tomotopy PRIVATE "${PYTHON3_STABLE_LIB}")
            target_include_directories(_tomotopy PRIVATE ${Python3_INCLUDE_DIRS})
        else()
            target_link_libraries(_tomotopy PRIVATE Python3::Module)
        endif()
    else()
        target_link_libraries(_tomotopy PRIVATE Python3::Module)
    endif()

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
            add_handler_module("_tomotopy_avx2" "avx2" "avx2")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        add_handler_module("_tomotopy_none" "arm64" "arm64")
    else()
        # x86_64 Linux/Windows: build all variants
        add_handler_module("_tomotopy_sse2" "sse2" "sse2")
        add_handler_module("_tomotopy_avx2" "avx2" "avx2")
        add_handler_module("_tomotopy_avx512" "avx512" "avx512")

        if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
            # Windows 32-bit needs 'none' for non-SSE2 CPUs
            add_handler_module("_tomotopy_none" "none" "none")
        endif()
    endif()
else()
    message(STATUS "Building single module for architecture: ${TARGET_ARCH}")
    add_handler_module("_tomotopy" "${TARGET_ARCH}" "${TOMOTOPY_ISA}")
endif()
