cmake_minimum_required(VERSION 3.0)

project(my_project)

# Find Boost library
find_package(fmt REQUIRED)

# Add executable
add_executable(my_exe main.cpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_link_libraries(my_exe
  Python3::Python
  Python3::Module)

find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  target_link_libraries(my_exe Python3::NumPy)
else()
  target_compile_definitions(my_exe INTERFACE WITHOUT_NUMPY)
endif()

target_link_libraries(my_exe fmt::fmt)
