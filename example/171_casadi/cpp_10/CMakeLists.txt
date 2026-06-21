cmake_minimum_required(VERSION 3.16)
project(PorkchopSim10 LANGUAGES CXX)

# C++20 verwenden
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Threading Building Blocks (TBB) suchen (erforderlich für Parallel STL in GCC)
find_package(TBB REQUIRED)

# Optimierungsflags für GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# Executable definieren
add_executable(porkchop_sim
    main.cpp
    lambert_solver.cpp
)

# TBB verlinken
target_link_libraries(porkchop_sim PRIVATE TBB::tbb)
