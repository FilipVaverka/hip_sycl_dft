#include "sycl_dft.h"
#include "sycl_dft_internal.h"

using namespace sycl::dft;

template<int BACKEND>
struct compute_forward_wrapper {
    template<typename... ArgsT>
    auto operator()(ArgsT&&... args) {
        return backend::compute_forward<BACKEND>(std::forward<ArgsT>(args)...);
    }
};

template<int BACKEND>
struct compute_backward_wrapper {
    template<typename... ArgsT>
    auto operator()(ArgsT&&... args) {
        return backend::compute_backward<BACKEND>(std::forward<ArgsT>(args)...);
    }
};

template<int BACKEND>
struct commit_wrapper {
    template<typename... ArgsT>
    void operator()(DFTDescriptorData_t::DFTBackendPtr_t &pBackend, ArgsT&&... args) {
        pBackend = backend::make_backend<BACKEND>();
        backend::commit<BACKEND>(pBackend, std::forward<ArgsT>(args)...);
    }
};

template<template<int> typename W, typename... ArgsT>
auto DispatchToBackend(sycl::queue &queue, ArgsT&&... args)
{
    int vendorId = queue.get_device().is_cpu() ? 0 : queue.get_device().get_info<sycl::info::device::vendor_id>();

    switch(vendorId)
    {
#ifdef SYCL_DFT_FFTW_BACKEND_ENABLED
    case DFT_BACKEND_OMP:
        return W<DFT_BACKEND_OMP>()(args...);
        break;
#endif
#ifdef SYCL_DFT_ROCFFT_BACKEND_ENABLED
    case DFT_BACKEND_HIP:
        return W<DFT_BACKEND_HIP>()(args...);
        break;
#endif
#ifdef SYCL_DFT_CUFFT_BACKEND_ENABLED
    case DFT_BACKEND_CUDA:
        return W<DFT_BACKEND_CUDA>()(args...);
        break;
#endif
    default:
        throw std::runtime_error("Unknown backend!");
        break;
    }
}

template<>
void DFTDescriptorData_t::set_value<config_value>(config_param param, const config_value &value) {
    switch(param)
    {
    case config_param::PLACEMENT: placement = value;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::get_value<config_value>(config_param param, config_value &value) {
    switch(param)
    {
    case config_param::PLACEMENT: value = placement;
        break;
    case config_param::COMMIT_STATUS: value = status;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::set_value<int>(config_param param, const int &value) {
    switch(param)
    {
    case config_param::NUMBER_OF_TRANSFORMS: count = value;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::get_value<int>(config_param param, int &value) {
    switch(param)
    {
    case config_param::NUMBER_OF_TRANSFORMS: value = count;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::set_value<DFTDescriptorData_t::WorkBufferPtr_t>(config_param param, const WorkBufferPtr_t &value) {
    switch(param)
    {
    case config_param::WORKSPACE: pWorkspace = value;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::get_value<DFTDescriptorData_t::WorkBufferPtr_t>(config_param param, WorkBufferPtr_t &value) {
    switch(param)
    {
    case config_param::WORKSPACE: value = pWorkspace;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::set_value<std::vector<size_t> >(config_param param, const std::vector<size_t> &value) {
    switch(param)
    {
    case config_param::INPUT_STRIDES: inputStrides = value;
        break;
    case config_param::OUTPUT_STRIDES: outputStrides = value;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::get_value<std::vector<size_t> >(config_param param, std::vector<size_t> &value) {
    switch(param)
    {
    case config_param::INPUT_STRIDES: value = inputStrides;
        break;
    case config_param::OUTPUT_STRIDES: value = outputStrides;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::set_value<size_t>(config_param param, const size_t &value) {
    switch(param)
    {
    case config_param::FWD_DISTANCE: forwardDist = value;
        break;
    case config_param::BWD_DISTANCE: backwardDist = value;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<>
void DFTDescriptorData_t::get_value<size_t>(config_param param, size_t &value) {
    switch(param)
    {
    case config_param::FWD_DISTANCE: value = forwardDist;
        break;
    case config_param::BWD_DISTANCE: value = backwardDist;
        break;
    default:
        throw std::runtime_error("Unsupported option!");
        break;
    }
}

template<sycl::dft::precision prec, sycl::dft::domain dom>
sycl::dft::descriptor<prec, dom>::descriptor(std::int64_t length)
    : handle(DFTDescriptorData_t::ToHandle(new DFTDescriptorData_t()))
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(handle);
    p->prec = prec;
    p->type = dom;

    p->dimensions.push_back(length);
    p->inputStrides.push_back(1);
    p->outputStrides.push_back(1);
}

template<sycl::dft::precision prec, sycl::dft::domain dom>
sycl::dft::descriptor<prec, dom>::descriptor(std::vector<std::int64_t> dimensions)
    : handle(DFTDescriptorData_t::ToHandle(new DFTDescriptorData_t()))
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(handle);
    p->prec = prec;
    p->type = dom;

    p->dimensions   = std::vector<size_t>(dimensions.begin(), dimensions.end());
    std::exclusive_scan(dimensions.begin(), dimensions.end(), std::back_inserter(p->inputStrides), 1, std::multiplies<>());
    std::exclusive_scan(dimensions.begin(), dimensions.end(), std::back_inserter(p->outputStrides), 1, std::multiplies<>());
}

template<sycl::dft::precision prec, sycl::dft::domain dom>
sycl::dft::descriptor<prec, dom>::~descriptor() 
{
    delete DFTDescriptorData_t::FromHandle(handle);
}

template<sycl::dft::precision prec, sycl::dft::domain dom>
template<typename T>
void sycl::dft::descriptor<prec, dom>::set_value(config_param param, const T &value) 
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(handle);
    p->set_value<T>(param, value);
}

template<sycl::dft::precision prec, sycl::dft::domain dom>
template<typename T>
void sycl::dft::descriptor<prec, dom>::get_value(config_param param, T &value) const
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(handle);
    p->get_value(param, value);
}

template<sycl::dft::precision prec, sycl::dft::domain dom>
void sycl::dft::descriptor<prec, dom>::commit(sycl::queue &queue) 
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(handle);
    p->pQueue = &queue;

    DispatchToBackend<commit_wrapper>(*p->pQueue, p->pBackend, *p);

    p->status = config_value::COMMITED;
}

template<typename descriptor_type, typename data_type, int D>
sycl::event sycl::dft::compute_forward(descriptor_type            &desc,
                                       sycl::buffer<data_type, D> &inout) 
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(desc.handle);

    return DispatchToBackend<compute_forward_wrapper>(*p->pQueue, p->pBackend, inout);
}

template<typename descriptor_type, typename input_type, typename output_type, int D>
sycl::event sycl::dft::compute_forward(descriptor_type              &desc,
                                       sycl::buffer<input_type, D>  &in,
                                       sycl::buffer<output_type, D> &out) 
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(desc.handle);

    return DispatchToBackend<compute_forward_wrapper>(*p->pQueue, p->pBackend, in, out);
}

template<typename descriptor_type, typename data_type, int D>
sycl::event sycl::dft::compute_backward(descriptor_type            &desc,
                                        sycl::buffer<data_type, D> &inout) 
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(desc.handle);

    return DispatchToBackend<compute_backward_wrapper>(*p->pQueue, p->pBackend, inout);
}

template<typename descriptor_type, typename input_type, typename output_type, int D>
sycl::event sycl::dft::compute_backward(descriptor_type              &desc,
                                        sycl::buffer<input_type, D>  &in,
                                        sycl::buffer<output_type, D> &out) 
{
    DFTDescriptorData_t *p = DFTDescriptorData_t::FromHandle(desc.handle);

    return DispatchToBackend<compute_backward_wrapper>(*p->pQueue, p->pBackend, in, out);
}

#define SYCL_DFT_DESC_INST(prec, dom) \
    template class sycl::dft::descriptor<precision::prec, domain::dom>; \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::get_value<config_value>(config_param param, config_value &value) const; \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::set_value<config_value>(config_param param, const config_value &value); \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::get_value<DFTDescriptorData_t::WorkBufferPtr_t>(config_param param, DFTDescriptorData_t::WorkBufferPtr_t &value) const; \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::set_value<DFTDescriptorData_t::WorkBufferPtr_t>(config_param param, const DFTDescriptorData_t::WorkBufferPtr_t &value); \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::get_value<std::vector<size_t>>(config_param param, std::vector<size_t> &value) const; \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::set_value<std::vector<size_t>>(config_param param, const std::vector<size_t> &value); \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::get_value<size_t>(config_param param, size_t &value) const; \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::set_value<size_t>(config_param param, const size_t &value); \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::get_value<int>(config_param param, int &value) const; \
    template void sycl::dft::descriptor<precision::prec, domain::dom>::set_value<int>(config_param param, const int &value)

#define SYCL_DFT_ROCFFT_INPL_INSTANTIATE(prec, dom, type, dim)                                              \
    template sycl::event sycl::dft::compute_forward<descriptor<precision::prec, domain::dom>, type, dim>(   \
        descriptor<precision::prec, domain::dom> &desc,                                                     \
        sycl::buffer<type, dim> &inout);                                                                    \
    template sycl::event sycl::dft::compute_backward<descriptor<precision::prec, domain::dom>, type, dim>(  \
        descriptor<precision::prec, domain::dom> &desc,                                                     \
        sycl::buffer<type, dim> &inout)

#define SYCL_DFT_ROCFFT_INPL_DIMS(prec, dom, type)        \
    SYCL_DFT_ROCFFT_INPL_INSTANTIATE(prec, dom, type, 1); \
    SYCL_DFT_ROCFFT_INPL_INSTANTIATE(prec, dom, type, 2); \
    SYCL_DFT_ROCFFT_INPL_INSTANTIATE(prec, dom, type, 3)

#define SYCL_DFT_ROCFFT_NINPL_INSTANTIATE(prec, dom, inType, outType, dim)                                            \
    template sycl::event sycl::dft::compute_forward<descriptor<precision::prec, domain::dom>, inType, outType, dim>(  \
        descriptor<precision::prec, domain::dom> &desc,                                                               \
        sycl::buffer<inType, dim>  &in,                                                                               \
        sycl::buffer<outType, dim> &out);                                                                             \
    template sycl::event sycl::dft::compute_backward<descriptor<precision::prec, domain::dom>, outType, inType, dim>( \
        descriptor<precision::prec, domain::dom> &desc,                                                               \
        sycl::buffer<outType, dim> &in,                                                                               \
        sycl::buffer<inType,  dim> &out)

#define SYCL_DFT_ROCFFT_NINPL_DIMS(prec, dom, inType, outType)        \
    SYCL_DFT_ROCFFT_NINPL_INSTANTIATE(prec, dom, inType, outType, 1); \
    SYCL_DFT_ROCFFT_NINPL_INSTANTIATE(prec, dom, inType, outType, 2); \
    SYCL_DFT_ROCFFT_NINPL_INSTANTIATE(prec, dom, inType, outType, 3)

SYCL_DFT_BACKEND_INST(sycl::dft::DFT_BACKEND_OMP, extern);
SYCL_DFT_BACKEND_INST(sycl::dft::DFT_BACKEND_HIP, extern);
SYCL_DFT_BACKEND_INST(sycl::dft::DFT_BACKEND_CUDA, extern);

SYCL_DFT_DESC_INST(SINGLE, REAL);
SYCL_DFT_DESC_INST(SINGLE, COMPLEX);

SYCL_DFT_ROCFFT_INPL_DIMS(SINGLE,    REAL, float);
SYCL_DFT_ROCFFT_INPL_DIMS(SINGLE, COMPLEX, sycl::float2);

SYCL_DFT_ROCFFT_NINPL_DIMS(SINGLE,    REAL,        float, sycl::float2);
SYCL_DFT_ROCFFT_NINPL_DIMS(SINGLE, COMPLEX, sycl::float2, sycl::float2);