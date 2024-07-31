cmake_minimum_required(VERSION 3.9)

project(tomotopy)

set ( CMAKE_CXX_STANDARD 17 )
set ( CMAKE_VERBOSE_MAKEFILE true )

set ( CMAKE_SKIP_RPATH ON )
set ( CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING "Minimum OS X deployment version" )

set ( EXT_TYPE "handler" CACHE STRING "dispatcher or handler" )
set ( TARGET_ARCH "" CACHE STRING "target architecture" )
set ( MODULE_NAME "_tomotopy" CACHE STRING "module name" )
set ( TOMOTOPY_ISA "none" CACHE STRING "target ISA" )
set ( NO_DEBUG_INFO OFF CACHE BOOL "no debug info" )
set ( TOMOTOPY_LANG "en" CACHE STRING "language" )

if(MSVC)
    if (TARGET_ARCH STREQUAL "none")
        set (ADDITIONAL_FLAGS "")
    elseif (TARGET_ARCH STREQUAL "sse2")
        set (ADDITIONAL_FLAGS "/arch:SSE2")
    elseif (TARGET_ARCH STREQUAL "avx")
        set (ADDITIONAL_FLAGS "/arch:AVX")
    elseif (TARGET_ARCH STREQUAL "avx2")
        set (ADDITIONAL_FLAGS "/arch:AVX2")
    endif()

    set ( CMAKE_C_FLAGS_DEBUG "-DDEBUG -DC_FLAGS -Zc:__cplusplus -Zi -Od ${ADDITIONAL_FLAGS}" )
    set ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" )

    set ( CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DRELEASE -DC_FLAGS -Zc:__cplusplus -O2 -Oi -Gy  ${ADDITIONAL_FLAGS}" )
    set ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" )

    set ( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -Zi /bigobj")
    set ( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    set ( CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" )
else()
    set (CPU_ARCH "x86_64")
    if (TARGET_ARCH STREQUAL "none")
        set (ADDITIONAL_FLAGS "")
    elseif (TARGET_ARCH STREQUAL "native")
        set (ADDITIONAL_FLAGS "-march=native")
    elseif (TARGET_ARCH STREQUAL "sse2")
        set (ADDITIONAL_FLAGS "-msse2")
    elseif (TARGET_ARCH STREQUAL "avx")
        set (ADDITIONAL_FLAGS "-mavx")
    elseif (TARGET_ARCH STREQUAL "avx2")
        set (ADDITIONAL_FLAGS "-mavx2")
    elseif (TARGET_ARCH STREQUAL "arm64")
        set (CPU_ARCH "arm64")
    endif()

    if (APPLE)
        message(STATUS "Apple Clang: use undefined=dynamic_lookup")
        set (CMAKE_OSX_ARCHITECTURES "${CPU_ARCH}")
        set (ADDITIONAL_FLAGS "${ADDITIONAL_FLAGS} -undefined dynamic_lookup")
        set (ADDITIONAL_LINKER_FLAGS "-stdlib=libc++")
    endif()

    set ( CMAKE_C_FLAGS_DEBUG "-DDEBUG -DC_FLAGS -g3 -O0 ${ADDITIONAL_FLAGS}" )
    set ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" )
    set ( CMAKE_EXE_LINKER_FLAGS_DEBUG "-DDEBUG -DLINKER_FLAGS ${ADDITIONAL_LINKER_FLAGS}" )

    set ( CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DRELEASE -DC_FLAGS -O3 ${ADDITIONAL_FLAGS}" )
    set ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" )
    set ( CMAKE_EXE_LINKER_FLAGS_RELEASE "-DRELEASE -DLINKER_FLAGS ${ADDITIONAL_LINKER_FLAGS}" )

    if (NO_DEBUG_INFO)
        set ( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g0")
        set ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g0")
        set ( CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g0")
        set ( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g0")
    endif()

    set ( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} -g3")
    set ( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    set ( CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${ADDITIONAL_LINKER_FLAGS}" )
endif()

SET (srcs 
)

if (EXT_TYPE STREQUAL "handler")
    file(GLOB cpp_files "src/python/handler/*.cpp" "src/Utils/*.cpp" "src/TopicModel/*.cpp" "src/Labeling/*.cpp")
    SET(srcs
        ${srcs}
        ${cpp_files}
    )
    add_definitions(-DMODULE_NAME=${MODULE_NAME})
    add_definitions(-DTOMOTOPY_ISA=${TOMOTOPY_ISA})
    if (TOMOTOPY_LANG STREQUAL "kr")
        add_definitions(-DDOC_KO=1)
    endif()
else()
    file(GLOB cpp_files "src/python/dispatcher/*.cpp")
    SET(srcs
        ${srcs}
        ${cpp_files}
    )
endif()

include_directories(${INCLUDE_DIRS}
)
link_directories(${LIBRARY_DIRS})

add_library("_tomotopy_target" SHARED
    ${srcs}
)

target_link_libraries("_tomotopy_target" PRIVATE 
  ${LIBRARIES}
)
