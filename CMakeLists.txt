cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-O3>")
project(GFSMProject LANGUAGES CXX CUDA)

add_executable(GFSM 
main.cu
ccsr.h
ccsr.cpp
HnGPU.cuh
HnGPU.cu
HnHelper.cuh
Environment.h)

set_property(TARGET GFSM PROPERTY CUDA_SEPARABLE_COMPILATION ON)