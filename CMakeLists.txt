cmake_minimum_required(VERSION 3.12)

if(DEFINED ENV{VCPKG_TARGET_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_TARGET_TRIPLET}" CACHE STRING "")
endif()
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  set(_VCPKG_INSTALLED_DIR ${CMAKE_SOURCE_DIR}/vcpkg_installed)
endif()

set(BUILD_GBENCH FALSE CACHE BOOL "Generate google benchmark executable")

project(abc CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# set the default build type if not set by user
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  endif()
endif()

# source list
set(SRC shared.h abc.cpp kde.h kde.cpp weighted.h weighted.cpp prior.h)

add_executable(abc ${SRC})
set_target_properties(abc PROPERTIES CXX_STANDARD 17 PREFIX "")

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS abc DESTINATION .)

if(BUILD_GBENCH)
  add_executable(gbench benchmark.cpp)
  set_target_properties(gbench PROPERTIES CXX_STANDARD 17 PREFIX "")
  if(WIN32)
    target_compile_options(gbench PRIVATE "/MD")
  endif()
  find_package(benchmark CONFIG REQUIRED)
  target_link_libraries(gbench PRIVATE benchmark::benchmark benchmark::benchmark_main)
  install(TARGETS gbench DESTINATION .)
endif()

# if(WIN32)
find_package(OpenMP)
target_link_libraries(abc PRIVATE OpenMP::OpenMP_CXX)
# endif()


# # Tiny Numpy IO library
# find_package(TinyNPY CONFIG REQUIRED)
# target_link_libraries(abc PRIVATE TinyNPY)

# EIGEN3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# FMT
find_package(fmt CONFIG REQUIRED)
target_link_libraries(abc PRIVATE fmt::fmt-header-only)

# Add native multithreading libraries
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(abc PRIVATE Threads::Threads)

# ↓↓↓ formatting with clang-format ↓↓↓
include(ClangFormat)
clangformat_setup(${SRC})
