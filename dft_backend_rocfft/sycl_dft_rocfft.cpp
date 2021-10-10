#include "sycl_dft_rocfft.h"

using namespace sycl::dft;


RocFFTBackend::RocFFTBackend()
    : pPlanForward(nullptr)
    , pPlanInverse(nullptr)
    , placement(rocfft_placement_inplace)
    , precision(rocfft_precision_single)
    , type(rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse)
    , pQueue(nullptr)
{
}

RocFFTBackend::~RocFFTBackend() 
{
    if(pPlanForward)
        rocfft_plan_destroy(pPlanForward);
    
    if(pPlanInverse)
        rocfft_plan_destroy(pPlanInverse);
}

static void CreatePlan(RocFFTBackend *p)
{
    hipDevice_t deviceId = sycl::get_native<sycl::backend::hip>(p->pQueue->get_device());
    if(hipSetDevice(deviceId) != hipSuccess)
        throw std::runtime_error("Failed to select HIP device!");

    const size_t inputDataSize  = p->inputLayout.size;
    const size_t outputDataSize = p->outputLayout.size;

    {
        rocfft_plan_description pPlanDesc;
        rocfft_plan_description_create(&pPlanDesc);
        rocfft_plan_description_set_data_layout(
            pPlanDesc,
            p->inputLayout.type, p->outputLayout.type,
            nullptr, nullptr,
            p->inputLayout.strides.size(), p->inputLayout.strides.data(), p->forwardDistance,
            p->outputLayout.strides.size(), p->outputLayout.strides.data(), p->backwardDistance);
        
        rocfft_plan_create(&p->pPlanForward, p->placement, 
            p->type.first, p->precision, 
            p->dimensions.size(), p->dimensions.data(), p->count, pPlanDesc);
        
        rocfft_plan_description_destroy(pPlanDesc);
    }

    {
        rocfft_plan_description pPlanDesc;
        rocfft_plan_description_create(&pPlanDesc);
        rocfft_plan_description_set_data_layout(
            pPlanDesc,
            p->outputLayout.type, p->inputLayout.type,
            nullptr, nullptr,
            p->outputLayout.strides.size(), p->outputLayout.strides.data(), p->backwardDistance,
            p->inputLayout.strides.size(), p->inputLayout.strides.data(), p->forwardDistance);
        
        rocfft_plan_create(&p->pPlanInverse, p->placement,
            p->type.second, p->precision,
            p->dimensions.size(), p->dimensions.data(), p->count, pPlanDesc);

        rocfft_plan_description_destroy(pPlanDesc);
    }

    size_t forwardWorkSize = 0;
    size_t inverseWorkSize = 0;
    rocfft_plan_get_work_buffer_size(p->pPlanForward, &forwardWorkSize);
    rocfft_plan_get_work_buffer_size(p->pPlanInverse, &inverseWorkSize);
    size_t workSize = std::max(forwardWorkSize, inverseWorkSize);

    if(!p->pWorkBuffer || p->pWorkBuffer->get_size() < workSize)
        p->pWorkBuffer = std::make_shared<sycl::buffer<uint8_t, 1> >(sycl::range(workSize));
}

static void CheckDFTConfig(const DFTDescriptorData_t &desc)
{
}

namespace sycl::dft::backend {

template<>
std::shared_ptr<void> make_backend<DFT_BACKEND_HIP>()
{
    return std::make_shared<RocFFTBackend>();
}

template<>
void commit<DFT_BACKEND_HIP>(std::shared_ptr<void> &backend, 
                             DFTDescriptorData_t   &desc)
{
    CheckDFTConfig(desc);

    std::shared_ptr<RocFFTBackend> p = RocFFTBackend::FromHandle(backend);

    p->pQueue      = desc.pQueue;
    p->pWorkBuffer = desc.pWorkspace;
    p->dimensions  = desc.dimensions;
    p->count       = desc.count;

    p->placement = (desc.placement == config_value::INPLACE) ? 
        rocfft_placement_inplace : rocfft_placement_notinplace;
    p->precision = (desc.prec == precision::SINGLE) ?
        rocfft_precision_single : rocfft_precision_double;
    
    p->inputLayout.strides  = desc.inputStrides;
    p->inputLayout.size     = desc.inputStrides.back() * desc.dimensions.back();
    p->outputLayout.strides = desc.outputStrides;
    p->outputLayout.size    = desc.outputStrides.back() * desc.dimensions.back();

    p->forwardDistance  = (desc.forwardDist  == 1) ? p->inputLayout.size  : desc.forwardDist;
    p->backwardDistance = (desc.backwardDist == 1) ? p->outputLayout.size : desc.backwardDist;

    if(desc.type == domain::REAL)
    {
        p->type = RocFFTBackend::TransformType_t(rocfft_transform_type_real_forward, rocfft_transform_type_real_inverse);
        p->inputLayout.type  = rocfft_array_type_real;
        p->outputLayout.type = rocfft_array_type_hermitian_interleaved;
    }
    else
    {
        p->type = RocFFTBackend::TransformType_t(rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse);
        p->inputLayout.type  = rocfft_array_type_complex_interleaved;
        p->outputLayout.type = rocfft_array_type_complex_interleaved;
    }

    CreatePlan(p.get());
}

template<int BACKEND, typename data_type, int D>
sycl::event compute_forward(std::shared_ptr<void>      &backend,
                            sycl::buffer<data_type, D> &inout)
{
    std::shared_ptr<RocFFTBackend> p = RocFFTBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto dataAcc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        auto tmpAcc  = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>();

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pData = h.get_native_mem<sycl::backend::hip>(dataAcc);
            void *pWork = h.get_native_mem<sycl::backend::hip>(tmpAcc);

            rocfft_execution_info pExecInfo;
            rocfft_execution_info_create(&pExecInfo);
            rocfft_execution_info_set_work_buffer(pExecInfo, pWork, tmpAcc.get_size());
            rocfft_execution_info_set_stream(pExecInfo, h.get_native_queue<sycl::backend::hip>());

            rocfft_execute(p->pPlanForward, &pData, &pData, pExecInfo);

            rocfft_execution_info_destroy(pExecInfo);
        });
    });
}

template<int BACKEND, typename input_type, typename output_type, int D>
sycl::event compute_forward(std::shared_ptr<void>        &backend,
                            sycl::buffer<input_type, D>  &in,
                            sycl::buffer<output_type, D> &out)
{
    std::shared_ptr<RocFFTBackend> p = RocFFTBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto inputAcc  = in.template get_access<sycl::access::mode::read>(cgh);
        auto outputAcc = out.template get_access<sycl::access::mode::discard_write>(cgh);
        auto tmpAcc    = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>(cgh);

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pInput  = h.get_native_mem<sycl::backend::hip>(inputAcc);
            void *pOutput = h.get_native_mem<sycl::backend::hip>(outputAcc);
            void *pWork   = h.get_native_mem<sycl::backend::hip>(tmpAcc);

            rocfft_execution_info pExecInfo;
            rocfft_execution_info_create(&pExecInfo);
            rocfft_execution_info_set_work_buffer(pExecInfo, pWork, tmpAcc.get_size());
            rocfft_execution_info_set_stream(pExecInfo, h.get_native_queue<sycl::backend::hip>());

            rocfft_execute(p->pPlanForward, &pInput, &pOutput, pExecInfo);

            rocfft_execution_info_destroy(pExecInfo);
        });
    });
}

template<int BACKEND, typename data_type, int D>
sycl::event compute_backward(std::shared_ptr<void>      &backend,
                             sycl::buffer<data_type, D> &inout)
{
    std::shared_ptr<RocFFTBackend> p = RocFFTBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto dataAcc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        auto tmpAcc  = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>();

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pData = h.get_native_mem<sycl::backend::hip>(dataAcc);
            void *pWork = h.get_native_mem<sycl::backend::hip>(tmpAcc);

            rocfft_execution_info pExecInfo;
            rocfft_execution_info_create(&pExecInfo);
            rocfft_execution_info_set_work_buffer(pExecInfo, pWork, tmpAcc.get_size());
            rocfft_execution_info_set_stream(pExecInfo, h.get_native_queue<sycl::backend::hip>());

            rocfft_execute(p->pPlanInverse, &pData, &pData, pExecInfo);

            rocfft_execution_info_destroy(pExecInfo);
        });
    });
}

template<int BACKEND, typename input_type, typename output_type, int D>
sycl::event compute_backward(std::shared_ptr<void>        &backend,
                             sycl::buffer<input_type, D>  &in,
                             sycl::buffer<output_type, D> &out)
{
    std::shared_ptr<RocFFTBackend> p = RocFFTBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto inputAcc  = in.template get_access<sycl::access::mode::read>(cgh);
        auto outputAcc = out.template get_access<sycl::access::mode::discard_write>(cgh);
        auto tmpAcc    = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>(cgh);

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pInput  = h.get_native_mem<sycl::backend::hip>(inputAcc);
            void *pOutput = h.get_native_mem<sycl::backend::hip>(outputAcc);
            void *pWork   = h.get_native_mem<sycl::backend::hip>(tmpAcc);

            rocfft_execution_info pExecInfo;
            rocfft_execution_info_create(&pExecInfo);
            rocfft_execution_info_set_work_buffer(pExecInfo, pWork, tmpAcc.get_size());
            rocfft_execution_info_set_stream(pExecInfo, h.get_native_queue<sycl::backend::hip>());

            rocfft_execute(p->pPlanInverse, &pInput, &pOutput, pExecInfo);

            rocfft_execution_info_destroy(pExecInfo);
        });
    });
}

}

SYCL_DFT_BACKEND_INST(DFT_BACKEND_HIP,);