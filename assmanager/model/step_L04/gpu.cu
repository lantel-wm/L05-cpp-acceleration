#include "gpu.hpp"

__global__ void calx_kernel(double* xens, double* zens_wrap, double* a, int ensemble_size, int model_size, int ss2, int smooth_steps) 
{
    int iens = blockIdx.x * blockDim.x + threadIdx.x;

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


void calx_gpu(double* xens, double* zens_wrap, double* a, int ensemble_size, int model_size, int ss2, int smooth_steps)
{
    int block_size = 256;
    int grid_size = (ensemble_size + block_size - 1) / block_size;

    calx_kernel<<<grid_size, block_size>>>(xens, zens_wrap, a, ensemble_size, model_size, ss2, smooth_steps);
}