cmake_minimum_required(VERSION 3.1)
project(sycl_dft)

option(SYCL_DFT_FFTW_BACKEND "Enable FFTW (CPU) backend for SYCL DFT" ON)
option(SYCL_DFT_ROCFFT_BACKEND "Enable rocFFT (GPU) backend for SYCL DFT" ON)
option(SYCL_DFT_CUFFT_BACKEND "Enable cuFFT (GPU) backend for SYCL DFT" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(PREPEND var prefix)
    set(listVar "")
    foreach(f ${ARGN})
        list(APPEND listVar "${prefix}/${f}")
    endforeach(f)
    set(${var} "${listVar}" PARENT_SCOPE)
endfunction(PREPEND)

file(GLOB_RECURSE HEADERS_LIST "src/*.h" "src/*.hpp")

set(SRCS_LIST)

add_subdirectory(src)

set(CMAKE_CXX_FLAGS_RELEASE -O3)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-I/opt/hipSYCL/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(${PROJECT_NAME} main.cpp ${SRCS_LIST} ${HEADERS_LIST})

if(${SYCL_DFT_FFTW_BACKEND})
    add_subdirectory(dft_backend_fftw)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SYCL_DFT_FFTW_BACKEND_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE dft_backend_fftw)
endif()

if(${SYCL_DFT_ROCFFT_BACKEND})
    add_subdirectory(dft_backend_rocfft)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SYCL_DFT_ROCFFT_BACKEND_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE dft_backend_rocfft)
endif()

if(${SYCL_DFT_CUFFT_BACKEND})
    add_subdirectory(dft_backend_cufft)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SYCL_DFT_CUFFT_BACKEND_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE dft_backend_cufft)
endif()

# if(CMAKE_EXPORT_COMPILE_COMMANDS)
#     set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
# endif()