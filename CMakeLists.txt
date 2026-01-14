cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(parallelSort MODULE temet/util/parallelSort.cpp)
install(TARGETS parallelSort DESTINATION ${SKBUILD_PROJECT_NAME})

