#include <algorithm>

#include "sycl_dft_fftw.h"

using namespace sycl::dft;


FFTWBackend::FFTWBackend()
    : pPlanForward(nullptr)
    , pPlanInverse(nullptr)
    , isInPlace(false)
    , pQueue(nullptr)
{
}

FFTWBackend::~FFTWBackend()
{
    if(pPlanForward)
        fftwf_destroy_plan(pPlanForward);
    
    if(pPlanInverse)
        fftwf_destroy_plan(pPlanInverse);
}

static void CreatePlan(FFTWBackend *p)
{
    fftwf_plan_with_nthreads(p->pQueue->get_device().get_info<sycl::info::device::max_compute_units>());

    /*size_t inputDataSize  = std::reduce(p->inputEmbed.begin(), p->inputEmbed.end(), 1, std::multiplies<>())   * p->inputStride;
    size_t outputDataSize = std::reduce(p->outputEmbed.begin(), p->outputEmbed.end(), 1, std::multiplies<>()) * p->outputStride;*/
    std::vector<fftwf_complex> tmpInput(p->forwardDistance * p->count), tmpOutput(p->backwardDistance * p->count);

    p->pPlanForward = fftwf_plan_many_dft(p->dimensions.size(), p->dimensions.data(), p->count,
        tmpInput.data(), p->inputEmbed.data(), p->inputStride, p->forwardDistance,
        p->isInPlace ? tmpInput.data() : tmpOutput.data(), p->outputEmbed.data(), p->outputStride, p->backwardDistance,
        FFTW_FORWARD, FFTW_MEASURE);
    
    p->pPlanInverse = fftwf_plan_many_dft(p->dimensions.size(), p->dimensions.data(), p->count, 
        tmpOutput.data(), p->outputEmbed.data(), p->outputStride, p->backwardDistance, 
        p->isInPlace ? tmpOutput.data() : tmpInput.data(), p->inputEmbed.data(), p->inputStride, p->forwardDistance, 
        FFTW_BACKWARD, FFTW_MEASURE);
}

static void CheckDFTConfig(const DFTDescriptorData_t &desc)
{
    if(desc.inputStrides.size() > 1)
        throw std::runtime_error("SYCL DFT[FFTW]: Input per-dimension strides not supported!");
    
    if(desc.outputStrides.size() > 1)
        throw std::runtime_error("SYCL DFT[FFTW]: Output per-dimension strides not supported!");
}

namespace sycl::dft::backend {

template<>
std::shared_ptr<void> make_backend<DFT_BACKEND_OMP>()
{
    return std::make_shared<FFTWBackend>();
}

template<>
void commit<DFT_BACKEND_OMP>(std::shared_ptr<void> &backend, 
                             DFTDescriptorData_t   &desc)
{
    std::shared_ptr<FFTWBackend> p = FFTWBackend::FromHandle(backend);

    p->pQueue = desc.pQueue;
    p->pWorkBuffer = desc.pWorkspace;
    p->count = desc.count;

    p->pWorkBuffer = std::make_shared<sycl::buffer<uint8_t, 1> >(sycl::range(1024));
    p->inputStride  = desc.inputStrides[0];
    p->outputStride = desc.outputStrides[0];
    
    p->dimensions.resize(desc.dimensions.size());
    std::reverse_copy(desc.dimensions.begin(), desc.dimensions.end(), p->dimensions.begin());

    {
        std::vector<size_t> embed;
        std::vector<size_t> extStrides = desc.inputStrides;

        if(desc.inputStrides.size() <= desc.dimensions.size())
            extStrides.push_back(desc.inputStrides.back() * desc.dimensions.back());

        for(size_t i = 0; i < extStrides.size() - 1; ++i)
        {
            embed.push_back(extStrides[i + 1] / extStrides[i]);
            std::cout << embed.back() << " ";
        }
        
        p->inputEmbed.resize(desc.dimensions.size());
        std::reverse_copy(embed.begin(), embed.end(), p->inputEmbed.begin());
        p->forwardDistance = (desc.forwardDist == 1) ? extStrides.back() : desc.forwardDist;
    }

    std::cout << std::endl;

    {
        std::vector<size_t> embed;
        std::vector<size_t> extStrides = desc.outputStrides;

        if(desc.outputStrides.size() <= desc.dimensions.size())
            extStrides.push_back(desc.outputStrides.back() * desc.dimensions.back());

        for(size_t i = 0; i < extStrides.size() - 1; ++i)
        {
            embed.push_back(extStrides[i + 1] / extStrides[i]);
        }
        
        p->outputEmbed.resize(desc.dimensions.size());
        std::reverse_copy(embed.begin(), embed.end(), p->outputEmbed.begin());
        p->backwardDistance = (desc.backwardDist == 1) ? extStrides.back() : desc.backwardDist;
    }
    
    CreatePlan(p.get());
}

template<int BACKEND, typename data_type, int D>
sycl::event compute_forward(std::shared_ptr<void>      &backend,
                            sycl::buffer<data_type, D> &inout)
{
    std::shared_ptr<FFTWBackend> p = FFTWBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto dataAcc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        auto tmpAcc  = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>();

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pData = dataAcc.get_pointer(); // h.get_native_mem<sycl::backend::omp>(dataAcc);
            void *pWork = tmpAcc.get_pointer();  // h.get_native_mem<sycl::backend::omp>(tmpAcc);

            fftwf_execute_dft(p->pPlanForward, reinterpret_cast<fftwf_complex *>(pData), 
                reinterpret_cast<fftwf_complex *>(pData));
        });
    });
}

template<int BACKEND, typename input_type, typename output_type, int D>
sycl::event compute_forward(std::shared_ptr<void>        &backend,
                            sycl::buffer<input_type, D>  &in,
                            sycl::buffer<output_type, D> &out)
{
    std::shared_ptr<FFTWBackend> p = FFTWBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto inputAcc  = in.template get_access<sycl::access::mode::read>(cgh);
        auto outputAcc = out.template get_access<sycl::access::mode::discard_write>(cgh);
        auto tmpAcc    = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>(cgh);

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pInput  = inputAcc.get_pointer(); // h.get_native_mem<sycl::backend::omp>(inputAcc);
            void *pOutput = outputAcc.get_pointer(); // h.get_native_mem<sycl::backend::omp>(outputAcc);
            void *pWork   = tmpAcc.get_pointer(); // h.get_native_mem<sycl::backend::omp>(tmpAcc);

            fftwf_execute_dft(p->pPlanForward, reinterpret_cast<fftwf_complex *>(pInput), 
                reinterpret_cast<fftwf_complex *>(pOutput));
        });
    });
}

template<int BACKEND, typename data_type, int D>
sycl::event compute_backward(std::shared_ptr<void>      &backend,
                             sycl::buffer<data_type, D> &inout)
{
    std::shared_ptr<FFTWBackend> p = FFTWBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto dataAcc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        auto tmpAcc  = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>();

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pData = dataAcc.get_pointer(); // h.get_native_mem<sycl::backend::omp>(dataAcc);
            void *pWork = tmpAcc.get_pointer();  // h.get_native_mem<sycl::backend::omp>(tmpAcc);

            fftwf_execute_dft(p->pPlanInverse, reinterpret_cast<fftwf_complex *>(pData), 
                reinterpret_cast<fftwf_complex *>(pData));
        });
    });
}

template<int BACKEND, typename input_type, typename output_type, int D>
sycl::event compute_backward(std::shared_ptr<void>        &backend,
                             sycl::buffer<input_type, D>  &in,
                             sycl::buffer<output_type, D> &out)
{
    std::shared_ptr<FFTWBackend> p = FFTWBackend::FromHandle(backend);

    return p->pQueue->submit([&](sycl::handler &cgh) {
        auto inputAcc  = in.template get_access<sycl::access::mode::read>(cgh);
        auto outputAcc = out.template get_access<sycl::access::mode::discard_write>(cgh);
        auto tmpAcc    = p->pWorkBuffer->get_access<sycl::access::mode::discard_read_write>(cgh);

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
            void *pInput  = inputAcc.get_pointer(); // h.get_native_mem<sycl::backend::omp>(inputAcc);
            void *pOutput = outputAcc.get_pointer(); // h.get_native_mem<sycl::backend::omp>(outputAcc);
            void *pWork   = tmpAcc.get_pointer(); // h.get_native_mem<sycl::backend::omp>(tmpAcc);

            fftwf_execute_dft(p->pPlanInverse, reinterpret_cast<fftwf_complex *>(pInput), 
                reinterpret_cast<fftwf_complex *>(pOutput));
        });
    });
}

}

SYCL_DFT_BACKEND_INST(DFT_BACKEND_OMP,);