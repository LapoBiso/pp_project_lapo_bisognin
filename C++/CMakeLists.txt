cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
project(pp_lapo_bisognin CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
execute_process(
        COMMAND brew --prefix
        OUTPUT_VARIABLE HOMEBREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${HOMEBREW_PREFIX}/opt/libomp/include)
link_directories(${HOMEBREW_PREFIX}/opt/libomp/lib)
include_directories(${HOMEBREW_PREFIX}/include)
link_directories(${HOMEBREW_PREFIX}/lib)
include_directories(include)
set(ARCH_FLAGS -mcpu=native)

set(BASE_OPTIMIZATION_FLAGS
        -O3
        ${ARCH_FLAGS}
        -ffast-math
        -DNDEBUG
)
add_executable(pp_lapo_bisognin
        src/main.cpp
        src/seqKmeans.cpp
        src/utilities.cpp
        src/parKmeans.cpp
)
target_compile_options(pp_lapo_bisognin PRIVATE
        ${BASE_OPTIMIZATION_FLAGS}
        ${VECTORIZATION_REPORT_FLAGS}
)
target_link_libraries(pp_lapo_bisognin PRIVATE omp profiler)
set_target_properties(pp_lapo_bisognin PROPERTIES
        BUILD_RPATH "${HOMEBREW_PREFIX}/lib"
        INSTALL_RPATH "${HOMEBREW_PREFIX}/lib"
)
