cmake_minimum_required(VERSION 3.0)

project(my_project)

# Find Boost library
find_package(fmt REQUIRED)

# Add executable
add_executable(my_exe main.cpp)

target_include_directories(my_exe PRIVATE /usr/include/python3.11)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#find_package(PythonLibs REQUIRED)
target_link_libraries(my_exe #INTERFACE
  Python3::Python
  Python3::Module
  fmt::fmt
)
target_link_libraries(my_exe ${PYTHON_LIBRARIES})
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  target_link_libraries(my_exe #INTERFACE
    Python3::NumPy
  )
else()
  target_compile_definitions(my_exe INTERFACE WITHOUT_NUMPY)
endif()

  
# Link with Boost libraries
#target_link_libraries(my_exe fmt::fmt)
