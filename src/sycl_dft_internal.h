#pragma once

#ifndef __SYCL_DFT_INTERNAL_H__
#define __SYCL_DFT_INTERNAL_H__

#include "sycl_dft.h"

namespace sycl::dft {

enum DFTBackends_t {
    DFT_BACKEND_OMP  = 0,
    DFT_BACKEND_HIP  = 1022,
    DFT_BACKEND_CUDA = 4318,
};

struct DFTDescriptorData_t
{
    typedef std::shared_ptr<sycl::buffer<uint8_t, 1> > WorkBufferPtr_t;
    typedef std::shared_ptr<void> DFTBackendPtr_t;
    
    DFTDescriptorData_t()
        : status(config_value::UNCOMMITED)
        , dimensions()
        , count(1)
        , placement(config_value::NOT_INPLACE)
        , prec(precision::SINGLE)
        , type(domain::COMPLEX)
        , pQueue(nullptr)
        , forwardDist(1)
        , backwardDist(1)
        , pWorkspace(nullptr)
        , pBackend(nullptr)
    {}

    static DFTDescriptorData_t *FromHandle(SYCL_DFT_DESC_HANLDE handle) { 
        return reinterpret_cast<DFTDescriptorData_t *>(handle);
    }

    static SYCL_DFT_DESC_HANLDE ToHandle(DFTDescriptorData_t *pDesc) {
        return reinterpret_cast<SYCL_DFT_DESC_HANLDE>(pDesc);
    }

    template<typename T>
    void set_value(config_param param, const T &value);

    template<typename T>
    void get_value(config_param param, T &value);

    config_value        status;
    std::vector<size_t> dimensions;
    size_t              count;

    config_value        placement;
    precision           prec;
    domain              type;
    sycl::queue         *pQueue;

    std::vector<size_t> inputStrides;
    std::vector<size_t> outputStrides;
    size_t              forwardDist;
    size_t              backwardDist;

    WorkBufferPtr_t pWorkspace;
    DFTBackendPtr_t pBackend;
};

namespace backend {

template<int BACKEND>
std::shared_ptr<void> make_backend();

template<int BACKEND>
void commit(std::shared_ptr<void> &backend, 
            DFTDescriptorData_t   &desc);

template<int BACKEND, typename data_type, int D>
sycl::event compute_forward(std::shared_ptr<void>      &backend,
                            sycl::buffer<data_type, D> &inout);

template<int BACKEND, typename input_type, typename output_type, int D>
sycl::event compute_forward(std::shared_ptr<void>        &backend,
                            sycl::buffer<input_type, D>  &in,
                            sycl::buffer<output_type, D> &out);

template<int BACKEND, typename data_type, int D>
sycl::event compute_backward(std::shared_ptr<void>      &backend,
                             sycl::buffer<data_type, D> &inout);

template<int BACKEND, typename input_type, typename output_type, int D>
sycl::event compute_backward(std::shared_ptr<void>        &backend,
                             sycl::buffer<input_type, D>  &in,
                             sycl::buffer<output_type, D> &out);

}

}

#define SYCL_DFT_BACKEND_INPL_INSTANTIATE(B, type, D, E) \
    E template sycl::event sycl::dft::backend::compute_forward<B, type, D>(std::shared_ptr<void>  &backend, sycl::buffer<type, D> &inout); \
    E template sycl::event sycl::dft::backend::compute_backward<B, type, D>(std::shared_ptr<void> &backend, sycl::buffer<type, D> &inout)

#define SYCL_DFT_BACKEND_INPL_DIMS(B, type, E)        \
    SYCL_DFT_BACKEND_INPL_INSTANTIATE(B, type, 1, E); \
    SYCL_DFT_BACKEND_INPL_INSTANTIATE(B, type, 2, E); \
    SYCL_DFT_BACKEND_INPL_INSTANTIATE(B, type, 3, E)

#define SYCL_DFT_BACKEND_NINPL_INSTANTIATE(B, input_type, output_type, D, E) \
    E template sycl::event sycl::dft::backend::compute_forward<B, input_type, output_type, D>(std::shared_ptr<void>  &backend, sycl::buffer<input_type, D> &in, sycl::buffer<output_type, D> &out); \
    E template sycl::event sycl::dft::backend::compute_backward<B, output_type, input_type, D>(std::shared_ptr<void> &backend, sycl::buffer<output_type, D> &in, sycl::buffer<input_type, D> &out)

#define SYCL_DFT_BACKEND_NINPL_DIMS(B, input_type, output_type, E) \
    SYCL_DFT_BACKEND_NINPL_INSTANTIATE(B, input_type, output_type, 1, E); \
    SYCL_DFT_BACKEND_NINPL_INSTANTIATE(B, input_type, output_type, 2, E); \
    SYCL_DFT_BACKEND_NINPL_INSTANTIATE(B, input_type, output_type, 3, E)


#define SYCL_DFT_BACKEND_INST(B, E) \
    SYCL_DFT_BACKEND_INPL_DIMS(B, float, E); \
    SYCL_DFT_BACKEND_INPL_DIMS(B, sycl::float2, E); \
    SYCL_DFT_BACKEND_NINPL_DIMS(B, float, sycl::float2, E); \
    SYCL_DFT_BACKEND_NINPL_DIMS(B, sycl::float2, sycl::float2, E); \

#endif // __SYCL_DFT_INTERNAL_H__