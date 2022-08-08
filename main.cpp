// ======== Copyright (c) 2022, Filip Vaverka, All rights reserved. ======== //
//
// Purpose:     HIP SYCL DFT library with multi-backend support
//
// $NoKeywords: $HipSyclDft $main.cpp
// $Date:       $2022-08-08
// ========================================================================= //

#include <iostream>
#include <vector>

#include "sycl_dft.h"


void test_dft_1d(sycl::queue &q, size_t n, size_t e, size_t s, size_t c);
void test_dft_2d(sycl::queue &q, size_t nx, size_t ny);
void test_dft_3d(sycl::queue &q, size_t nx, size_t ny, size_t nz);

void test_dft_r2c_1d(sycl::queue &q, size_t n);

void PrintDeviceInfo(const sycl::device &d, float error) {
    std::cout << d.get_info<sycl::info::device::name>() 
              << " (" << d.get_info<sycl::info::device::vendor>() << "): "
              << error << std::endl;
}

int main(int argc, char *argv[])
{
    // sycl::default_selector selector;
    sycl::queue q_cpu{sycl::cpu_selector()};
    sycl::queue q_gpu{sycl::gpu_selector()};

    std::cout << "1D: R2C" << std::endl;
    test_dft_r2c_1d(q_cpu, 256);
    test_dft_r2c_1d(q_gpu, 256);

    std::cout << "1D: C2C" << std::endl;
    test_dft_1d(q_cpu, 256, 512, 1, 4);
    test_dft_1d(q_gpu, 256, 512, 1, 4);

    std::cout << "2D: C2C" << std::endl;
    test_dft_2d(q_cpu, 128, 256);
    test_dft_2d(q_gpu, 128, 256);

    std::cout << "3D: C2C" << std::endl;
    test_dft_3d(q_cpu, 64, 128, 256);
    test_dft_3d(q_gpu, 64, 128, 256);

    return 0;
}

void test_dft_1d(sycl::queue &q, size_t n, size_t e, size_t s, size_t c)
{
    sycl::dft::descriptor<sycl::dft::precision::SINGLE, sycl::dft::domain::COMPLEX> d(n);
    d.set_value<sycl::dft::config_param::NUMBER_OF_TRANSFORMS>(int(c));
    d.set_value<sycl::dft::config_param::INPUT_STRIDES>({s});
    d.set_value<sycl::dft::config_param::OUTPUT_STRIDES>({s});
    d.set_value<sycl::dft::config_param::FWD_DISTANCE>(e);
    d.set_value<sycl::dft::config_param::BWD_DISTANCE>(e);

    d.commit(q);

    std::vector<sycl::float2> input(e*s*c);
    std::vector<sycl::float2> output(e*s*c);

    for(size_t j = 0; j < c; ++j)
    {
        for(size_t i = 0; i < n; ++i)
        {
            float x    = i * ((2.0f * M_PI) / float(n));
            input[j*e*s + i*s] = sycl::float2{cosf(x), 0.0f};
        }
    }

    {
        auto inputBuffer  = sycl::buffer<sycl::float2, 1>(input.data(), sycl::range<1>(input.size()));
        auto tmpBuffer    = sycl::buffer<sycl::float2, 1>(sycl::range<1>(input.size()));
        auto outputBuffer = sycl::buffer<sycl::float2, 1>(output.data(), sycl::range<1>(output.size()));

        sycl::dft::compute_forward(d, inputBuffer, tmpBuffer);
        sycl::dft::compute_backward(d, tmpBuffer, outputBuffer);
    }

    float maxError = 0.0f;
    for(size_t j = 0; j < c; ++j)
    {
        for(size_t i = 0; i < n; ++i)
        {
            sycl::float2 diff = input[j*e*s + i*s] - output[j*e*s + i*s] * (1.0f / float(n));
            maxError = std::max(maxError, std::fabs(diff.x()));
        }
    }

    PrintDeviceInfo(q.get_device(), maxError);
}

void test_dft_2d(sycl::queue &q, size_t nx, size_t ny)
{
    sycl::dft::descriptor<sycl::dft::precision::SINGLE, sycl::dft::domain::COMPLEX> d({int64_t(nx), int64_t(ny)});
    d.commit(q);

    std::vector<sycl::float2> input(nx*ny);
    std::vector<sycl::float2> output(nx*ny);

    for(size_t i = 0; i < ny; ++i)
    {
        for(size_t j = 0; j < nx; ++j)
        {
            float x  = j * ((2.0f * M_PI) / float(nx));
            input[i*nx + j] = sycl::float2{cosf(x), 0.0f};
        }
    }

    {
        auto inputBuffer  = sycl::buffer<sycl::float2, 1>(input.data(), sycl::range<1>(input.size()));
        auto tmpBuffer    = sycl::buffer<sycl::float2, 1>(sycl::range<1>(nx*ny));
        auto outputBuffer = sycl::buffer<sycl::float2, 1>(output.data(), sycl::range<1>(output.size()));

        sycl::dft::compute_forward(d, inputBuffer, tmpBuffer);
        sycl::dft::compute_backward(d, tmpBuffer, outputBuffer);
    }

    float maxError = 0.0f;
    for(size_t i = 0; i < nx*ny; ++i)
    {
        sycl::float2 diff = input[i] - output[i] * (1.0f / float(nx*ny));
        maxError = std::max(maxError, std::fabs(diff.x()));
    }

    PrintDeviceInfo(q.get_device(), maxError);
}

void test_dft_3d(sycl::queue &q, size_t nx, size_t ny, size_t nz)
{
    sycl::dft::descriptor<sycl::dft::precision::SINGLE, sycl::dft::domain::COMPLEX> d({int64_t(nx), int64_t(ny), int64_t(nz)});
    d.commit(q);

    std::vector<sycl::float2> input(nx*ny*nz);
    std::vector<sycl::float2> output(nx*ny*nz);

    for(size_t i = 0; i < nz; ++i)
    {
        for(size_t j = 0; j < ny; ++j)
        {
            for(size_t k = 0; k < nx; ++k)
            {
                float x  = k * ((2.0f * M_PI) / float(nx));
                input[i*ny*nx + j*nx + k] = sycl::float2{cosf(x), 0.0f};
            }
        }
    }

    {
        auto inputBuffer  = sycl::buffer<sycl::float2, 1>(input.data(), sycl::range<1>(input.size()));
        auto tmpBuffer    = sycl::buffer<sycl::float2, 1>(sycl::range<1>(nx*ny*nz));
        auto outputBuffer = sycl::buffer<sycl::float2, 1>(output.data(), sycl::range<1>(output.size()));

        sycl::dft::compute_forward(d, inputBuffer, tmpBuffer);
        sycl::dft::compute_backward(d, tmpBuffer, outputBuffer);
    }

    float maxError = 0.0f;
    for(size_t i = 0; i < nx*ny*nz; ++i)
    {
        sycl::float2 diff = input[i] - output[i] * (1.0f / float(nx*ny*nz));
        maxError = std::max(maxError, std::fabs(diff.x()));
    }

    PrintDeviceInfo(q.get_device(), maxError);
}

void test_dft_r2c_1d(sycl::queue &q, size_t n)
{
    sycl::dft::descriptor<sycl::dft::precision::SINGLE, sycl::dft::domain::REAL> d(n);

    d.commit(q);

    std::vector<float> input(n);
    std::vector<float> output(n);

    for(size_t i = 0; i < n; ++i)
    {
        float x  = i * ((2.0f * M_PI) / float(n));
        input[i] = cosf(x);
    }

    {
        auto inputBuffer  = sycl::buffer<float, 1>(input.data(), sycl::range<1>(input.size()));
        auto tmpBuffer    = sycl::buffer<sycl::float2, 1>(sycl::range<1>(input.size()));
        auto outputBuffer = sycl::buffer<float, 1>(output.data(), sycl::range<1>(output.size()));

        sycl::dft::compute_forward(d, inputBuffer, tmpBuffer);
        sycl::dft::compute_backward(d, tmpBuffer, outputBuffer);
    }

    float maxError = 0.0f;
    for(size_t i = 0; i < n; ++i)
    {
        float diff = input[i] - output[i] * (1.0f / float(n));
        maxError = std::max(maxError, std::fabs(diff));
    }

    PrintDeviceInfo(q.get_device(), maxError);
}