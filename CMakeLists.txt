cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)

project(Smart1 LANGUAGES CXX CUDA)
enable_language(CUDA)



find_package(CUDA 11.0 REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(src Smart1/)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

include_directories(${src})

add_executable(Smart ${src}kernel.cu ${src}Network.cu ${src}Matrix.cu ${src}Layer.cu)

