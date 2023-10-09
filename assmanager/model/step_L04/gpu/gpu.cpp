// cuda extension for step_L04
// coded by Zhiyu Zhao, 2023/10/09
// Email: zyzh@smail.nju.edu.cn
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK(call)                                                          \
    {                                                                         \
        const cudaError_t error = call;                                       \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                     \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                          \
        }                                                                     \
    }

namespace py = pybind11;

#ifdef __cplusplus
extern "C"
#endif
void run_calx_kernel(double* xens, double* zens_wrap, double* a, int ensemble_size, int model_size, int ss2, int smooth_steps);
void run_calw_kernel(double* wxens, double* xens_wrap, int ensemble_size, int model_size, int K, int K4, int H);

py::array_t<double> calx(py::array_t<double> xens_np, py::array_t<double> zens_wrap_np, py::array_t<double> a_np, int model_size, int ss2, int smooth_steps)
{
    // xens_np: (ensemble_size, model_size) e.g.: (40, 960)
    // use cuda kernel function to parallelize ensemble dimension
    py::buffer_info xens_info = xens_np.request();
    py::buffer_info zens_wrap_info = zens_wrap_np.request();
    py::buffer_info a_info = a_np.request();

    double *xens = (double *)xens_info.ptr; // (40, 960)
    double *zens_wrap = (double *)zens_wrap_info.ptr;
    double *a = (double *)a_info.ptr;

    // auto ensemble_size = xens_info.shape[0];
    // auto model_size = xens_info.shape[1];

    double *d_xens, *d_zens_wrap, *d_a;
    CHECK(cudaMalloc((void **)&d_xens, xens_info.size * sizeof(double)));
    CHECK(cudaMalloc((void **)&d_zens_wrap, zens_wrap_info.size * sizeof(double)));
    CHECK(cudaMalloc((void **)&d_a, a_info.size * sizeof(double)));

    CHECK(cudaMemcpy(d_xens, xens, xens_info.size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_zens_wrap, zens_wrap, zens_wrap_info.size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_a, a, a_info.size * sizeof(double), cudaMemcpyHostToDevice));

    run_calx_kernel(d_xens, d_zens_wrap, d_a, xens_info.shape[0], xens_info.shape[1], ss2, smooth_steps);

    CHECK(cudaMemcpy(xens, d_xens, xens_info.size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_xens));
    CHECK(cudaFree(d_zens_wrap));
    CHECK(cudaFree(d_a));

    return xens_np;
}


py::array_t<double> calw(py::array_t<double> wxens_np, py::array_t<double> xens_wrap_np, int K, int K4, int H, int model_size)
{
    py::buffer_info wxens_info = wxens_np.request();
    py::buffer_info xens_wrap_info = xens_wrap_np.request();

    double *wxens = (double *)wxens_info.ptr;
    double *xens_wrap = (double *)xens_wrap_info.ptr;

    double *d_wxens, *d_xens_wrap;
    CHECK(cudaMalloc((void **)&d_wxens, wxens_info.size * sizeof(double)));
    CHECK(cudaMalloc((void **)&d_xens_wrap, xens_wrap_info.size * sizeof(double)));

    CHECK(cudaMemcpy(d_wxens, wxens, wxens_info.size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_xens_wrap, xens_wrap, xens_wrap_info.size * sizeof(double), cudaMemcpyHostToDevice));

    run_calw_kernel(d_wxens, d_xens_wrap, wxens_info.shape[0], wxens_info.shape[1], K, K4, H);

    CHECK(cudaMemcpy(wxens, d_wxens, wxens_info.size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_wxens));
    CHECK(cudaFree(d_xens_wrap));

    return wxens_np;
}


PYBIND11_MODULE(gpu, m)
{
    m.doc() = "step_L04 cpp extension, gpu version";
    m.def("calx", &calx, "calx function");
    m.def("calw", &calw, "calw function");
    // m.def("caldz", &caldz, "caldz function");
}