cmake_minimum_required(VERSION 3.22)
project(DLsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wextra)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(simdjson
  URL https://github.com/simdjson/simdjson/archive/refs/tags/v3.13.0.tar.gz)
set(SIMDJSON_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(SIMDJSON_DEVELOPER_MODE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(simdjson)

FetchContent_Declare(yaml-cpp
  URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

FetchContent_Declare(eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${eigen_SOURCE_DIR})

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

file(GLOB_RECURSE DLSIM_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(FILTER DLSIM_CORE_SOURCES EXCLUDE REGEX "/src/cli/")
add_library(dlsim_core STATIC ${DLSIM_CORE_SOURCES})
target_include_directories(dlsim_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dlsim_core PUBLIC simdjson yaml-cpp::yaml-cpp eigen Threads::Threads OpenSSL::Crypto)
target_compile_definitions(dlsim_core PUBLIC DLSIM_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

file(GLOB DLSIM_CLI_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/cli/*.cpp)
add_executable(dlsim ${DLSIM_CLI_SOURCES})
target_link_libraries(dlsim PRIVATE dlsim_core)

option(DLSIM_BUILD_IMPORT "Build dlsim-import (needs Arrow/Parquet C++)" ON)
if(DLSIM_BUILD_IMPORT AND EXISTS ${CMAKE_SOURCE_DIR}/tools/import/CMakeLists.txt)
  add_subdirectory(tools/import)
endif()

enable_testing()
file(GLOB DLSIM_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*_test.cpp)
foreach(test_src ${DLSIM_TESTS})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE dlsim_core)
  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()
