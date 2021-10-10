#include <cuda.h>
#include <cuda_runtime.h>

#include "sycl_dft_cufft.h"

using namespace sycl::dft;

CuFFTBackend::CuFFTBackend()
    : pPlanForward(CUFFT_INVALID_PLAN)
    , pPlanInverse(CUFFT_INVALID_PLAN)
    , type(CUFFT_C2C)
    , pQueue(nullptr)
{

}

CuFFTBackend::~CuFFTBackend()
{
    if(pPlanForward != CUFFT_INVALID_PLAN)
        cufftDestroy(pPlanForward);
    
    if(pPlanInverse != CUFFT_INVALID_PLAN)
        cufftDestroy(pPlanInverse);
}

void CreatePlan(CuFFTBackend *p)
{
    int deviceId = sycl::get_native<sycl::backend::cuda>(p->pQueue->get_device());
    if(cudaSetDevice(deviceId) != cudaSuccess)
        throw std::runtime_error("Failed to select CUDA device!");
    
    cufftPlanMany(&p->pPlanForward, p->dimensions.size(), p->dimensions.data(),
        p->inputEmbed.data(), p->inputStride, p->forwardDistance,
        p->outputEmbed.data(), p->outputStride, p->backwardDistance,
        p->type, p->count);
    
    cufftPlanMany(&p->pPlanInverse, p->dimensions.size(), p->dimensions.data(),
        p->outputEmbed.data(), p->outputStride, p->backwardDistance,
        p->inputEmbed.data(), p->inputStride, p->forwardDistance,
        p->type, p->count);

    size_t forwardWorkSize = 0;
    size_t inverseWorkSize = 0;
    cufftGetSize(p->pPlanForward, &forwardWorkSize);
    cufftGetSize(p->pPlanInverse, &inverseWorkSize);
    size_t workSize = std::max(forwardWorkSize, inverseWorkSize);

    if(!p->pWorkBuffer || p->pWorkBuffer->get_size() < workSize)
        p->pWorkBuffer = std::make_shared<sycl::buffer<uint8_t, 1> >(sycl::range(workSize));
}

namespace sycl::dft::backend {

template<>
std::shared_ptr<void> make_backend<DFT_BACKEND_CUDA>()
{
    return std::make_shared<CuFFTBackend>();
}

template<>
void commit<DFT_BACKEND_CUDA>(std::shared_ptr<void> &backend, 
                             DFTDescriptorData_t   &desc)
{
    std::shared_ptr<CuFFTBackend> p = CuFFTBackend::FromHandle(backend);

    p->pQueue = desc.pQueue;
    p->pWorkBuffer = desc.pWorkspace;
    p->count = desc.count;
    
    p->inputStride = desc.inputStrides[0];
    p->outputStride = desc.outputStrides[0];

    p->dimensions.resize(desc.dimensions.size());
    std::reverse_copy(desc.dimensions.begin(), desc.dimensions.end(), p->dimensions.begin());

    {
        std::vector<size_t> embed;
        std::vector<size_t> extStrides = desc.inputStrides;
        extStrides.push_back(desc.inputStrides.back() * desc.dimensions.back());

        for(size_t i = 0; i < extStrides.size() - 1; ++i)
            embed.push_back(extStrides[i + 1] / extStrides[i]);
        
        p->inputEmbed.resize(desc.dimensions.size());
        std::reverse_copy(embed.begin(), embed.end(), p->inputEmbed.begin());
        p->forwardDistance = (desc.forwardDist == 1) ? extStrides.back() : desc.forwardDist;
    }

    {
        std::vector<size_t> embed;
        std::vector<size_t> extStrides = desc.outputStrides;
        extStrides.push_back(desc.outputStrides.back() * desc.dimensions.back());

        for(size_t i = 0; i < extStrides.size() - 1; ++i)
            embed.push_back(extStrides[i + 1] / extStrides[i]);
        
        p->outputEmbed.resize(desc.dimensions.size());
        std::reverse_copy(embed.begin(), embed.end(), p->outputEmbed.begin());
        p->backwardDistance = (desc.backwardDist == 1) ? extStrides.back() : desc.backwardDist;
    }

    /*p->pQueue = desc.pQueue;
    p->pWorkBuffer = desc.pWorkspace;

    p->placement = (desc.placement == config_value::INPLACE) ? 
        rocfft_placement_inplace : rocfft_placement_notinplace;
    p->precision = (desc.prec == precision::SINGLE) ?
        rocfft_precision_single : rocfft_precision_double;
    p->type = (desc.type == domain::REAL) ?
        RocFFTBackend::TransformType_t(rocfft_transform_type_real_forward, rocfft_transform_type_real_inverse) :
        RocFFTBackend::TransformType_t(rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse);
    
    p->dimensions = desc.dimensions;
    p->count = desc.count;*/

    CreatePlan(p.get());
}

template<int BACKEND, typename data_type, int D>
sycl::event compute_forward(std::shared_ptr<void>      &backend,
                            sycl::buffer<data_type, D> &inout)
{
    std::shared_ptr<CuFFTBackend> p = CuFFTBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto dataAcc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        auto tmpAcc  = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>();

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pData = h.get_native_mem<sycl::backend::cuda>(dataAcc);
            void *pWork = h.get_native_mem<sycl::backend::cuda>(tmpAcc);

            cufftSetWorkArea(p->pPlanForward, pWork);
            cufftSetStream(p->pPlanForward, h.get_native_queue<sycl::backend::cuda>());

            cufftExecC2C(p->pPlanForward, 
                reinterpret_cast<cufftComplex *>(pData), 
                reinterpret_cast<cufftComplex *>(pData), CUFFT_FORWARD);

            /*rocfft_execution_info pExecInfo;
            rocfft_execution_info_create(&pExecInfo);
            rocfft_execution_info_set_work_buffer(pExecInfo, pWork, tmpAcc.get_size());
            rocfft_execution_info_set_stream(pExecInfo, h.get_native_queue<sycl::backend::cuda>());

            rocfft_execute(p->pPlanForward, &pData, &pData, pExecInfo);

            rocfft_execution_info_destroy(pExecInfo);*/
        });
    });
}

template<int BACKEND, typename input_type, typename output_type, int D>
sycl::event compute_forward(std::shared_ptr<void>        &backend,
                            sycl::buffer<input_type, D>  &in,
                            sycl::buffer<output_type, D> &out)
{
    std::shared_ptr<CuFFTBackend> p = CuFFTBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto inputAcc  = in.template get_access<sycl::access::mode::read>(cgh);
        auto outputAcc = out.template get_access<sycl::access::mode::discard_write>(cgh);
        auto tmpAcc    = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>(cgh);

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pInput  = h.get_native_mem<sycl::backend::cuda>(inputAcc);
            void *pOutput = h.get_native_mem<sycl::backend::cuda>(outputAcc);
            void *pWork   = h.get_native_mem<sycl::backend::cuda>(tmpAcc);

            cufftSetWorkArea(p->pPlanForward, pWork);
            cufftSetStream(p->pPlanForward, h.get_native_queue<sycl::backend::cuda>());

            cufftExecC2C(p->pPlanForward, 
                reinterpret_cast<cufftComplex *>(pInput), 
                reinterpret_cast<cufftComplex *>(pOutput), CUFFT_FORWARD);

            /*rocfft_execution_info pExecInfo;
            rocfft_execution_info_create(&pExecInfo);
            rocfft_execution_info_set_work_buffer(pExecInfo, pWork, tmpAcc.get_size());
            rocfft_execution_info_set_stream(pExecInfo, h.get_native_queue<sycl::backend::cuda>());

            rocfft_execute(p->pPlanForward, &pInput, &pOutput, pExecInfo);

            rocfft_execution_info_destroy(pExecInfo);*/
        });
    });
}

template<int BACKEND, typename data_type, int D>
sycl::event compute_backward(std::shared_ptr<void>      &backend,
                             sycl::buffer<data_type, D> &inout)
{
    std::shared_ptr<CuFFTBackend> p = CuFFTBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto dataAcc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        auto tmpAcc  = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>();

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pData = h.get_native_mem<sycl::backend::cuda>(dataAcc);
            void *pWork = h.get_native_mem<sycl::backend::cuda>(tmpAcc);

            cufftSetWorkArea(p->pPlanForward, pWork);
            cufftSetStream(p->pPlanForward, h.get_native_queue<sycl::backend::cuda>());

            cufftExecC2C(p->pPlanForward, 
                reinterpret_cast<cufftComplex *>(pData), 
                reinterpret_cast<cufftComplex *>(pData), CUFFT_INVERSE);

            /*rocfft_execution_info pExecInfo;
            rocfft_execution_info_create(&pExecInfo);
            rocfft_execution_info_set_work_buffer(pExecInfo, pWork, tmpAcc.get_size());
            rocfft_execution_info_set_stream(pExecInfo, h.get_native_queue<sycl::backend::cuda>());

            rocfft_execute(p->pPlanInverse, &pData, &pData, pExecInfo);

            rocfft_execution_info_destroy(pExecInfo);*/
        });
    });
}

template<int BACKEND, typename input_type, typename output_type, int D>
sycl::event compute_backward(std::shared_ptr<void>        &backend,
                             sycl::buffer<input_type, D>  &in,
                             sycl::buffer<output_type, D> &out)
{
    std::shared_ptr<CuFFTBackend> p = CuFFTBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto inputAcc  = in.template get_access<sycl::access::mode::read>(cgh);
        auto outputAcc = out.template get_access<sycl::access::mode::discard_write>(cgh);
        auto tmpAcc    = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>(cgh);

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pInput  = h.get_native_mem<sycl::backend::cuda>(inputAcc);
            void *pOutput = h.get_native_mem<sycl::backend::cuda>(outputAcc);
            void *pWork   = h.get_native_mem<sycl::backend::cuda>(tmpAcc);

            cufftSetWorkArea(p->pPlanForward, pWork);
            cufftSetStream(p->pPlanForward, h.get_native_queue<sycl::backend::cuda>());

            cufftExecC2C(p->pPlanForward, 
                reinterpret_cast<cufftComplex *>(pInput), 
                reinterpret_cast<cufftComplex *>(pOutput), CUFFT_FORWARD);

            /*rocfft_execution_info pExecInfo;
            rocfft_execution_info_create(&pExecInfo);
            rocfft_execution_info_set_work_buffer(pExecInfo, pWork, tmpAcc.get_size());
            rocfft_execution_info_set_stream(pExecInfo, h.get_native_queue<sycl::backend::hip>());

            rocfft_execute(p->pPlanInverse, &pInput, &pOutput, pExecInfo);

            rocfft_execution_info_destroy(pExecInfo);*/
        });
    });
}

}

SYCL_DFT_BACKEND_INST(DFT_BACKEND_CUDA,);