file(GLOB_RECURSE HEADERS_LIST "*.h" "*.hpp")

set(SRCS_LIST
    sycl_dft_cufft.cpp
)

add_library(dft_backend_cufft EXCLUDE_FROM_ALL ${SRCS_LIST} ${HEADERS_LIST})
target_compile_options(dft_backend_cufft PRIVATE -I/usr/local/cuda/include --hipsycl-targets=${SYCL_DFT_CUFFT_ARCHS})
target_include_directories(dft_backend_cufft PRIVATE /usr/local/cuda/include)
target_link_libraries(dft_backend_cufft PRIVATE cufft)