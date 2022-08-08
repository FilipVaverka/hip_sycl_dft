# Multi-backend HIP SYCL DFT library

Multi-backend HIPSYCL DFT library with interface inspired by DFT implemented in OneMKL/DPC++.

The library supports CUDA (cuFFT), HIP/ROCm (rocFFT) and OpenMP (FFTW) backends, which can be used simultaneously.

## Build
```console
mkdir build
cd build
CXX=syclcc cmake -DSYCL_DFT_TARGET_ARCHS="omp;hip:gfx906,gfx1030;cuda:sm_75" ..
make
```
Commands above will build HIP DFT tests for OpenMP, HIP (gfx906 and gfx1030) and CUDA (sm_75).

Backends are then enabled based on these specified HIPSYCL targets (ie. "omp" enables FFTW backend or "cuda:sm_75" enables cuFFT backend compiled for "sm_75").
```console
./sycl_dft 
1D: R2C
256 
hipSYCL OpenMP host device (the hipSYCL project): 1.78814e-07
AMD Radeon VII (AMD): 4.17233e-07
1D: C2C
256 
hipSYCL OpenMP host device (the hipSYCL project): 2.38419e-07
AMD Radeon VII (AMD): 1.78814e-07
2D: C2C
128 256 
hipSYCL OpenMP host device (the hipSYCL project): 1.78814e-07
AMD Radeon VII (AMD): 1.78814e-07
3D: C2C
64 128 256 
hipSYCL OpenMP host device (the hipSYCL project): 1.19209e-07
AMD Radeon VII (AMD): 1.78814e-07
```
Runs simple 1D, 2D and 3D transforms on CPU (OpenMP) and one of avaliable GPU targets.