cmake_minimum_required(VERSION 3.10.0)

project(neuralnetwork C CXX)

set(neuralnetwork_includes
    include/neuralnetwork.hpp
)

add_library(neuralnetwork STATIC ${neuralnetwork_includes})
target_include_directories(neuralnetwork
	PUBLIC include
	PUBLIC gsl/include
)
set_property(TARGET neuralnetwork PROPERTY CXX_STANDARD 17)
set_target_properties(neuralnetwork PROPERTIES LINKER_LANGUAGE CXX)

set(NEURALNETWORK_GENERATE_EXAMPLE OFF CACHE BOOL "When set to ON, the neural network example target will be generated")

if (NEURALNETWORK_GENERATE_EXAMPLE)
    add_executable(neuralnetwork_demo demo.cpp)
    set_property(TARGET neuralnetwork_demo PROPERTY CXX_STANDARD 17)
    target_link_libraries(neuralnetwork_demo PUBLIC neuralnetwork)

    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT neuralnetwork_demo)
endif()
