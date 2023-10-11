#include <cuda_runtime.h>

__global__ void calx_kernel(double* xens, double* zens_wrap, double* a, int ensemble_size, int model_size, int ss2, int smooth_steps) 
{
    int iens = blockIdx.x * blockDim.x + threadIdx.x;

    if (iens >= ensemble_size)
    {
        return;
    }

    double *x = xens + iens * model_size;
    double *zwrap = zens_wrap + iens * model_size;

    for (int i = ss2; i < ss2 + model_size; i++)
    {
        x[i - ss2] = a[0] * zwrap[i + 1 - (- smooth_steps)] / 2.00;
        for (int j = -smooth_steps + 1; j < smooth_steps; j++)
        {
            x[i - ss2] = x[i - ss2] + a[j + smooth_steps] * zwrap[i + 1 - j];
        }
        x[i - ss2] = x[i - ss2] + a[2 * smooth_steps] * zwrap[i + 1 - smooth_steps] / 2.00;
    }
}


__global__ void calw(double* wxens, double* xens_wrap, int ensemble_size, int model_size, int K, int K4, int H) 
{
    int iens = blockIdx.x * blockDim.x + threadIdx.x;

    if (iens >= ensemble_size)
    {
        return;
    }

    double *wx = wxens + iens * model_size;
    double *xwrap = xens_wrap + iens * model_size;

    for (int i = K4; i < K4 + model_size; i++)
    {
        wx[i] = xwrap[i - (-H)] / 2.00;
        for (int j = -H + 1; j < H; j++)
            wx[i] = wx[i] + xwrap[i - j];

        wx[i] = wx[i] + xwrap[i - H] / 2.00;
        wx[i] = wx[i] / K;
    }
}

     
__global__ void caldz(double* wxens, double* xens_wrap, double* dzens, double* yens_wrap,
          int ensemble_size, int model_size, double space_time_scale, double sts2, double coupling, 
          double forcing, int K, int K2, int K4, int H, int model_number) 
{
    int iens = blockIdx.x * blockDim.x + threadIdx.x;

    if (iens >= ensemble_size)
    {
        return;
    }

    double *wx = wxens + iens * model_size;
    double *xwrap = xens_wrap + iens * model_size;
    double *dz = dzens + iens * model_size;
    double *ywrap = yens_wrap + iens * model_size;

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
}


#ifdef __cplusplus
extern "C"
#endif

void run_calx_kernel(double* xens, double* zens_wrap, double* a, int ensemble_size, int model_size, int ss2, int smooth_steps)
{
    int block_size = 256;
    int grid_size = (ensemble_size + block_size - 1) / block_size;

    calx_kernel<<<grid_size, block_size>>>(xens, zens_wrap, a, ensemble_size, model_size, ss2, smooth_steps);
}


void run_calw_kernel(double* wxens, double* xens_wrap, int ensemble_size, int model_size, int K, int K4, int H)
{
    int block_size = 256;
    int grid_size = (ensemble_size + block_size - 1) / block_size;

    calw<<<grid_size, block_size>>>(wxens, xens_wrap, ensemble_size, model_size, K, K4, H);
}


void run_caldz_kernel(double* wxens, double* xens_wrap, double* dzens, double* yens_wrap,
          int ensemble_size, int model_size, double space_time_scale, double sts2, double coupling, 
          double forcing, int K, int K2, int K4, int H, int model_number)
{
    int block_size = 256;
    int grid_size = (ensemble_size + block_size - 1) / block_size;

    caldz<<<grid_size, block_size>>>(wxens, xens_wrap, dzens, yens_wrap,
          ensemble_size, model_size, space_time_scale, sts2, coupling, 
          forcing, K, K2, K4, H, model_number);
}