#pragma once

#ifndef __SYCL_DFT_CUFFT_H__
#define __SYCL_DFT_CUFFT_H__

#include <cufft.h>

#define SYCL_EXT_HIPSYCL_BACKEND_CUDA
#include <SYCL/sycl.hpp>

#include "sycl_dft.h"
#include "sycl_dft_internal.h"

struct CuFFTBackend
{
    CuFFTBackend();
    virtual ~CuFFTBackend();

    static std::shared_ptr<CuFFTBackend> FromHandle(std::shared_ptr<void> pBackendHandle) {
        return std::reinterpret_pointer_cast<CuFFTBackend>(pBackendHandle);
    }

    std::vector<int> dimensions;
    int              count;
    int              inputStride;
    int              outputStride;
    std::vector<int> inputEmbed;
    std::vector<int> outputEmbed;
    int              forwardDistance;
    int              backwardDistamce;

    cufftHandle pPlanForward;
    cufftHandle pPlanInverse;

    cufftType   type;

    sycl::queue *pQueue;
    sycl::dft::DFTDescriptorData_t::WorkBufferPtr_t pWorkBuffer;
};


#endif // __SYCL_DFT_CUFFT_H__