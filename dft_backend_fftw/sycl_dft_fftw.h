#pragma once

#ifndef __SYCL_DFT_FFTW_H__
#define __SYCL_DFT_FFTW_H__

#include <fftw3.h>

#define SYCL_EXT_HIPSYCL_BACKEND_OMP
#include <SYCL/sycl.hpp>

#include "sycl_dft.h"
#include "sycl_dft_internal.h"

struct FFTWBackend
{
    FFTWBackend();
    virtual ~FFTWBackend();

    static std::shared_ptr<FFTWBackend> FromHandle(std::shared_ptr<void> pBackendHandle) {
        return std::reinterpret_pointer_cast<FFTWBackend>(pBackendHandle);
    }

    std::vector<int> dimensions;
    int              count;
    int              inputStride;
    int              outputStride;
    std::vector<int> inputEmbed;
    std::vector<int> outputEmbed;
    int              forwardDistance;
    int              backwardDistance;

    fftwf_plan pPlanForward;
    fftwf_plan pPlanInverse;
    bool       isInPlace;

    sycl::queue *pQueue;
    sycl::dft::DFTDescriptorData_t::WorkBufferPtr_t pWorkBuffer;
};

#endif // __SYCL_DFT_FFTW_H__