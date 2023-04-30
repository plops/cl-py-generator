cmake_minimum_required(VERSION 3.16)

project(OFDM_SchmidlCox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(FFTW3 REQUIRED)

# Define the executable
add_executable(${PROJECT_NAME} main.cpp)

# Set compiler flags for debug build
set(DEBUG_FLAGS "-g3 -Wall -Wextra -Wconversion -Wdouble-promotion -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -fsanitize=undefined -fsanitize-trap=undefined -fsanitize=address")

target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_FLAGS}>")

# Link the libraries
target_link_libraries(${PROJECT_NAME} fftw3)
