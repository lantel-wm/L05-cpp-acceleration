// cpp extension for step_L04
// coded by Zhiyu Zhao, 2023/10/09
// Email: zyzh@smail.nju.edu.cn
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

py::array_t<double> calx(py::array_t<double> x_np, py::array_t<double> zwrap_np, py::array_t<double> a_np, int model_size, int ss2, int smooth_steps)
{
    py::buffer_info x_info = x_np.request();
    py::buffer_info zwrap_info = zwrap_np.request();
    py::buffer_info a_info = a_np.request();

    double *x = (double *)x_info.ptr;
    double *zwrap = (double *)zwrap_info.ptr;
    double *a = (double *)a_info.ptr;

    for (int i = ss2; i < ss2 + model_size; i++)
    {
        x[i - ss2] = a[0] * zwrap[i + 1 - (- smooth_steps)] / 2.00;
        for (int j = -smooth_steps + 1; j < smooth_steps; j++)
        {
            x[i - ss2] = x[i - ss2] + a[j + smooth_steps] * zwrap[i + 1 - j];
        }
        x[i - ss2] = x[i - ss2] + a[2 * smooth_steps] * zwrap[i + 1 - smooth_steps] / 2.00;
    }

    return x_np;
}

py::array_t<double> calw(py::array_t<double> wx_np, py::array_t<double> xwrap_np, int K, int K4, int H, int model_size)
{
    py::buffer_info wx_info = wx_np.request();
    py::buffer_info xwrap_info = xwrap_np.request();

    double *wx = (double *)wx_info.ptr;
    double *xwrap = (double *)xwrap_info.ptr;

    for (int i = K4; i < K4 + model_size; i++)
    {
        wx[i] = xwrap[i - (-H)] / 2.00;
        for (int j = -H + 1; j < H; j++)
        {
            wx[i] = wx[i] + xwrap[i - j];
        }
        wx[i] = wx[i] + xwrap[i - H] / 2.00;
        wx[i] = wx[i] / K;
    }

    return wx_np;
}

py::array_t<double> caldz(py::array_t<double> wx_np, py::array_t<double> xwrap_np, py::array_t<double> dz_np, py::array_t<double> ywrap_np,
    double space_time_scale, double sts2, double coupling, double forcing, int K, int K2, int K4, int H, int model_size, int model_number)
{
    py::buffer_info wx_info = wx_np.request();
    py::buffer_info xwrap_info = xwrap_np.request();
    py::buffer_info dz_info = dz_np.request();
    py::buffer_info ywrap_info = ywrap_np.request();

    double *wx = (double *)wx_info.ptr;
    double *xwrap = (double *)xwrap_info.ptr;
    double *dz = (double *)dz_info.ptr;
    double *ywrap = (double *)ywrap_info.ptr;

    for (int i = K4; i < K4 + model_size; i++)
    {
        double xx = wx[i - K + (-H)] * xwrap[i + K + (-H)] / 2.00;
        for (int j = -H + 1; j < H; j++)
        {
            xx = xx + wx[i - K + j] * xwrap[i + K + j];
        }
        xx = xx + wx[i - K + H] * xwrap[i + K + H] / 2.00;
        xx = - wx[i - K2] * wx[i - K] + xx / K;

        if (model_number == 3)
        {
            dz[i - K4] = xx + sts2 * (- ywrap[i - 2] * ywrap[i - 1] + ywrap[i - 1] * ywrap[i + 1])
                            + coupling * (- ywrap[i - 2] * xwrap[i - 1] + ywrap[i - 1] * xwrap[i + 1]) - xwrap[i]
                            - space_time_scale * ywrap[i] + forcing;
        }
        else // model II
        {
            dz[i - K4] = xx - xwrap[i] + forcing;
        }
    }

    return dz_np;
}

PYBIND11_MODULE(cpu, m)
{
    m.doc() = "step_L04 cpp extension, cpu version";
    m.def("calx", &calx, "calx function");
    m.def("calw", &calw, "calw function");
    m.def("caldz", &caldz, "caldz function");
}