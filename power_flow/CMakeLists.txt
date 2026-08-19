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

# Standalone demo: solve a textbook 2x2 linear equation with the SAME SparseLU
# used by the power-flow solver. Exists to validate the general-equation
# wrapper with a trivial case before running 2000-bus systems.
add_executable(test_eq2x2
    src/sparse_matrix.cpp
    src/lu_decomposition.cpp
    src/test_eq2x2.cpp
)
target_include_directories(test_eq2x2 PRIVATE include)

# Benchmark: 100x100 pentadiagonal system -> 3-solver correctness+speed
# comparison (our SparseLU vs hand-rolled dense LU vs Eigen::SparseLU).
find_package(Eigen3 QUIET)
add_executable(benchmark_100x100
    src/sparse_matrix.cpp
    src/lu_decomposition.cpp
    src/benchmark_100x100.cpp
)
target_include_directories(benchmark_100x100 PRIVATE include)
if(Eigen3_FOUND)
    target_link_libraries(benchmark_100x100 PRIVATE Eigen3::Eigen)
    message(STATUS "Eigen3: found, benchmark_100x100 will use Eigen::SparseLU")
else()
    # Fallback: the Eigen headers installed by apt are NOT exposed via
    # Eigen3::Eigen CMake target, so we add the include dir manually.
    target_include_directories(benchmark_100x100 PRIVATE /usr/include/eigen3)
    message(STATUS "Eigen3: using system header path /usr/include/eigen3")
endif()

# Topology tracing via MATRIX operations (no DFS/BFS recursion).
# Uses the project SparseMatrix for the adjacency CSR and SpMV frontier
# expansion; correctness is cross-validated vs a DFS+BFS reference.
add_executable(topology_matrix_trace
    src/sparse_matrix.cpp
    src/topology_matrix_trace.cpp
)
target_include_directories(topology_matrix_trace PRIVATE include)

# Topology scaling benchmark: BFS (queue) vs Matrix (SpMV waves) across
# n = 10, 100, 1000, 10000, 20000, 50000, 100000 nodes.
add_executable(topology_benchmark_scaling
    src/sparse_matrix.cpp
    src/topology_benchmark_scaling.cpp
)
target_include_directories(topology_benchmark_scaling PRIVATE include)

# Enumerate all VALID radial operating modes for a 2-feeder network with
# N sectionalizers per feeder and K tie-switches between feeders.
# Compares traditional BFS/backtracking + union-find vs batched
# 64-way bit-parallel (SWAR) AND-OR transitive closure (matrix method).
add_executable(topology_operating_modes
    src/topology_operating_modes.cpp
)
target_include_directories(topology_operating_modes PRIVATE include)

# Small debug helper for the modes enumerator (brute force reference).
add_executable(debug_modes
    src/debug_modes.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(power_flow PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP: enabled")
else()
    message(STATUS "OpenMP: not found, building serial fallback")
endif()

add_executable(debug_compare src/debug_compare.cpp)
