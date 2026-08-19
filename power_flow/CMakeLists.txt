cmake_minimum_required(VERSION 3.10)
project(power_flow CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimized release flags. -march=native enables SIMD; fall back if unsupported.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

find_package(Threads REQUIRED)
find_package(OpenMP QUIET)

add_executable(power_flow
    src/sparse_matrix.cpp
    src/lu_decomposition.cpp
    src/power_system.cpp
    src/newton_raphson.cpp
    src/fast_decoupled.cpp
    src/main.cpp
)

target_include_directories(power_flow PRIVATE include)

target_link_libraries(power_flow PRIVATE Threads::Threads)

if(OpenMP_CXX_FOUND)
    target_link_libraries(power_flow PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP: enabled")
else()
    message(STATUS "OpenMP: not found, building serial fallback")
endif()
