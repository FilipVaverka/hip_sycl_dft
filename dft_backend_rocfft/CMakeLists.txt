file(GLOB_RECURSE HEADERS_LIST "*.h" "*.hpp")

set(SRCS_LIST
    sycl_dft_rocfft.cpp
)

find_package(hip REQUIRED)
find_package(rocfft REQUIRED)

add_library(dft_backend_rocfft EXCLUDE_FROM_ALL ${SRCS_LIST} ${HEADERS_LIST})
target_compile_options(dft_backend_rocfft PRIVATE -I${ROCFFT_INCLUDE_DIRS} -I${HIP_INCLUDE_DIRS} --hipsycl-targets=${SYCL_DFT_ROCFFT_ARCHS})
target_include_directories(dft_backend_rocfft PRIVATE ${ROCFFT_INCLUDE_DIRS})
target_link_libraries(dft_backend_rocfft PRIVATE roc::rocfft)