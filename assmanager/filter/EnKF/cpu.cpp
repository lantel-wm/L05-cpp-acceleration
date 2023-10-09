// cpp extension for EnKF filter
// coded by Zhiyu Zhao, 2023/10/09
// Email: zyzh@smail.nju.edu.cn
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


double mean1d(double *x, int n)
{
    double xmean = 0.00;
    for (int i = 0; i < n; i++)
    {
        xmean = xmean + x[i];
    }
    xmean = xmean / n;
    return xmean;
}


double* mean2d(double *x, int nrows, int ncols, int axis=0)
{
    double *xmean;
    if (axis == 0)
    {
        xmean = new double[ncols];
        for (int i = 0; i < ncols; i++)
        {
            xmean[i] = 0.00;
            for (int j = 0; j < nrows; j++)
            {
                xmean[i] = xmean[i] + x[j * ncols + i];
            }
            xmean[i] = xmean[i] / nrows;
        }
    }
    else if (axis == 1)
    {
        xmean = new double[nrows];
        for (int i = 0; i < nrows; i++)
        {
            xmean[i] = 0.00;
            for (int j = 0; j < ncols; j++)
            {
                xmean[i] = xmean[i] + x[i * ncols + j];
            }
            xmean[i] = xmean[i] / ncols;
        }
    }
    return xmean;
}

double* substration(double *x, double *xmean, int nrows, int ncols)
{
    // x: 40 x n
    // xmean: 1 x n
    double *xprime = new double[nrows * ncols];
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            xprime[i * ncols + j] = x[i * ncols + j] - xmean[j];
        }
    }
    return xprime;
}

double* get_hxens(double *zens, int nrows1, int ncols1, double *Hk, int nrows2, int ncols2, int iobs)
{
    int obs_loc = 0;
    for (int igrid = 0; igrid < ncols1; igrid ++)
    {
        // Hk[iobs, igrid]
        if (Hk[iobs * ncols2 + igrid] == 1)
        {
            obs_loc = igrid;
            break;
        }
    }
    double *hxens = new double[nrows1];
    for (int iens = 0; iens < nrows1; iens++)
    {
        hxens[iens] = zens[iens * ncols1 + obs_loc];
    }
    return hxens;
}

double dot_product(double *x, double *y, int n)
{
    double dot = 0.00;
    for (int i = 0; i < n; i++)
    {
        dot = dot + x[i] * y[i];
    }
    return dot;
}

double* mat_mul_vec(double *A, int nrows, int ncols, double *x, int n)
{
    // (40, n) * (40, 1)
    double *y = new double[ncols];
    for (int i = 0; i < ncols; i++)
    {
        y[i] = 0.00;
        for (int j = 0; j < nrows; j++)
        {
            y[i] = y[i] + A[j * ncols + i] * x[j];
        }
    }
    return y;
}


double* vec_mul_vec_elewise(double *x, double *y, int n)
{
    double *z = new double[n];
    for (int i = 0; i < n; i++)
    {
        z[i] = x[i] * y[i];
    }
    return z;
}


double* vec_mul(double *x, int m, double *y, int n)
{
    // (m, 1) * (1, n)
    double *A = new double[m * n];
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; i++)
        {
            A[i * n + j] = x[i] * y[j];
        }
    }
    return A;
}


py::array_t<double> serial_update(py::array_t<double> zens_np, py::array_t<double> zobs_np, py::array_t<double> Hk_np, py::array_t<double> CMat_np, int nobsgrid, bool localize, double obs_error_var)
{
    py::buffer_info zens_info = zens_np.request();
    py::buffer_info zobs_info = zobs_np.request();
    py::buffer_info Hk_info = Hk_np.request();
    py::buffer_info CMat_info = CMat_np.request();

    double *zens = (double *)zens_info.ptr;
    double *zobs = (double *)zobs_info.ptr;
    double *Hk = (double *)Hk_info.ptr; // 240 x 960
    double *CMat = (double *)CMat_info.ptr; // 240 x 960

    auto xmean_np = py::array_t<double>(zens_info.shape[1]); // 1 x n
    auto xprime_np = py::array_t<double>(zens_info.size); // 40 x n
    xprime_np.resize({zens_info.shape[0], zens_info.shape[1]});
    auto hxens_np = py::array_t<double>(zens_info.shape[0] * 1); // 40 x 1
    double hxmean;
    auto hxprime_np = py::array_t<double>(zens_info.shape[0] * 1); // 40 x 1
    double hpbht;
    auto pbht_np = py::array_t<double>(zens_info.shape[0] * 1); // n x 1
    auto kfgain_np = py::array_t<double>(zens_info.shape[0] * 1); // n x 1

    double *xmean = (double *)xmean_np.request().ptr;
    double *xprime = (double *)xprime_np.request().ptr;
    double *hxens = (double *)hxens_np.request().ptr;
    double *hxprime = (double *)hxprime_np.request().ptr;
    double *pbht = (double *)pbht_np.request().ptr;
    double *kfgain = (double *)kfgain_np.request().ptr;

    double rn = 1.00 / (zens_info.shape[0] - 1);
    for (int iobs = 0; iobs < nobsgrid; iobs++)
    {
        xmean = mean2d(zens, zens_info.shape[0], zens_info.shape[1], 0);
        xprime = substration(zens, xmean, zens_info.shape[0], zens_info.shape[1]);
        hxens = get_hxens(zens, zens_info.shape[0], zens_info.shape[1], Hk, Hk_info.shape[0], Hk_info.shape[1], iobs);
        hxmean = mean1d(hxens, zens_info.shape[0]);

        for (int iens = 0; iens < zens_info.shape[0]; iens++)
        {
            hxprime[iens] = hxens[iens] - hxmean;
        }

        hpbht = dot_product(hxprime, hxprime, zens_info.shape[0]) * rn;
        pbht = mat_mul_vec(xprime, zens_info.shape[0], zens_info.shape[1], hxprime, zens_info.shape[0]);

        for (int igrid = 0; igrid < zens_info.shape[0]; igrid++)
        {
            kfgain[igrid] = pbht[igrid] / (hpbht + obs_error_var);
        }

        if (localize)
        {
            kfgain = vec_mul_vec_elewise(kfgain, CMat + iobs * CMat_info.shape[1], CMat_info.shape[1]);
        }

        for (int iens = 0; iens < zens_info.shape[0]; iens++)
        {
            for (int igrid = 0; igrid < zens_info.shape[1]; igrid++)
            {
                zens[iens * zens_info.shape[1] + igrid] = zens[iens * zens_info.shape[1] + igrid] + kfgain[igrid] * (zobs[iens * zobs_info.shape[1] + iobs] - hxens[iens]);
            }
        }

    }    

    return zens_np;
}


PYBIND11_MODULE(cpu, m)
{
    m.doc() = "EnKF filter cpp extension, cpu version";
    m.def("serial_update", &serial_update, "serial update");
}