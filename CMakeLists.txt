cmake_minimum_required(VERSION 3.10.0)

project(neuralnetwork C CXX)

set(neuralnetwork_includes
    include/neuralnetwork.hpp
)

set(neuralnetwork_srcs
	src/neuralnetwork.cpp
)

add_library(neuralnetwork STATIC ${neuralnetwork_includes} ${neuralnetwork_srcs})
target_include_directories(neuralnetwork
	PUBLIC include
	PUBLIC gsl/include
)
set_property(TARGET neuralnetwork PROPERTY CXX_STANDARD 17)
set_target_properties(neuralnetwork PROPERTIES LINKER_LANGUAGE CXX)

add_executable(neuralnetwork_demo demo.cpp)
set_property(TARGET neuralnetwork_demo PROPERTY CXX_STANDARD 17)
target_link_libraries(neuralnetwork_demo PUBLIC neuralnetwork)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT neuralnetwork_demo)