cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP)

pybind11_add_module(parallelSort MODULE temet/util/parallelSort.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(parallelSort PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS util/parallelSort DESTINATION ${SKBUILD_PROJECT_NAME})
