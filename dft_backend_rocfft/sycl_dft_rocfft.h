#pragma once

#ifndef __SYCL_DFT_ROCFFT_H__
#define __SYCL_DFT_ROCFFT_H__

#include <rocfft.h>

#define SYCL_EXT_HIPSYCL_BACKEND_HIP
#include "sycl_dft.h"
#include "sycl_dft_internal.h"

struct RocFFTBackend
{
    typedef std::pair<rocfft_transform_type, rocfft_transform_type> TransformType_t;

    struct DataLayout_t {
        std::vector<size_t> strides;
        rocfft_array_type   type;
        size_t              size;
    };

    RocFFTBackend();
    virtual ~RocFFTBackend();

    static std::shared_ptr<RocFFTBackend> FromHandle(std::shared_ptr<void> pBackendHandle) {
        return std::reinterpret_pointer_cast<RocFFTBackend>(pBackendHandle);
    }

    std::vector<size_t> dimensions;
    size_t              count;

    DataLayout_t inputLayout;
    DataLayout_t outputLayout;
    size_t       forwardDistance;
    size_t       backwardDistance;

    rocfft_plan pPlanForward;
    rocfft_plan pPlanInverse;

    rocfft_result_placement placement;
    rocfft_precision        precision;
    TransformType_t         type;

    sycl::queue             *pQueue;
    sycl::dft::DFTDescriptorData_t::WorkBufferPtr_t pWorkBuffer;
};

#endif // __SYCL_DFT_ROCFFT_H__