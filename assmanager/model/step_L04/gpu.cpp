// cuda extension for step_L04
// coded by Zhiyu Zhao, 2023/10/09
// Email: zyzh@smail.nju.edu.cn
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gpu.hpp"

namespace py = pybind11;

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

    calx_gpu(d_xens, d_zens_wrap, d_a, xens_info.shape[0], xens_info.shape[1], ss2, smooth_steps);

    CHECK(cudaMemcpy(xens, d_xens, xens_info.size * sizeof(double), cudaMemcpyDeviceToHost));

    return xens_np;
}

PYBIND11_MODULE(gpu, m)
{
    m.doc() = "step_L04 cpp extension, gpu version";
    m.def("calx", &calx, "calx function");
    // m.def("calw", &calw, "calw function");
    // m.def("caldz", &caldz, "caldz function");
}