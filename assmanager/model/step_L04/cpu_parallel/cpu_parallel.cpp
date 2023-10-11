#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>

namespace py = pybind11;

void calx(double *x, double *zwrap, double *a, int model_size, int ss2, int smooth_steps)
{
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

void calw(double* wx, double* xwrap, int K, int K4, int H, int model_size)
{
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
}

void caldz(double* dz, double* wx, double* xwrap, double* ywrap,
    double space_time_scale, double sts2, double coupling, double forcing, int K, int K2, int K4, int H, int model_size, int model_number)
{
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


void z2xy(double *x, double *y, double *z, double*a, int model_size, double smooth_steps, double ss2)
{
    double *zwrap = new double[model_size + 2 * static_cast<int>(ss2) + 1];
    for (int i = model_size - static_cast<int>(ss2) - 1, j = 0; i < model_size; i++, j++) zwrap[j] = z[i];
    for (int i = 0, j = static_cast<int>(ss2) + 1; i < model_size; i++, j++) zwrap[j] = z[i];
    for (int i = 0, j = static_cast<int>(ss2) + model_size + 1; i < static_cast<int>(ss2); i++, j++) zwrap[j] = z[i];

    calx(x, zwrap, a, model_size, ss2, smooth_steps);

    for(int i = 0; i < model_size; i++)
    {
        y[i] = z[i] - x[i];
    }
    delete[] zwrap;
}


void comp_dt_L04(double* dz, double* z, double *a, int model_size, double smooth_steps, double ss2, double space_time_scale, double sts2, double coupling, double forcing, int K, int K2, int K4, int H, int model_number)
{
    double *x = new double[model_size];
    double *y = new double[model_size];
    // memset(x, 0, model_size);
    // memset(y, 0, model_size);
    for (int i = 0; i < model_size; i++) x[i] = 0;
    for (int i = 0; i < model_size; i++) y[i] = 0;

    if (model_number == 3)
    {
        z2xy(x, y, z, a, model_size, smooth_steps, ss2);
    }
    else if (model_number == 2)
    {
        // memcpy(x, z, model_size);
        for (int i = 0; i < model_size; i++) x[i] = z[i];
        // memset(y, 0, model_size);
        for (int i = 0; i < model_size; i++) y[i] = 0;
    }
    else // raise error
    {
        std::cout << "model_number must be 2 or 3" << std::endl;
        exit(1);
    }

    // Deal with  # cyclic boundary# conditions using buffers
    // Fill the xwrap and ywrap buffers
    double *xwrap = new double[model_size + 2 * K4];
    double *ywrap = new double[model_size + 2 * K4];
    for (int i = model_size - K4, j = 0; i < model_size; i++, j++) {xwrap[j] = x[i]; ywrap[j] = y[i];}
    for (int i = 0, j = K4; i < model_size; i++, j++) {xwrap[j] = x[i]; ywrap[j] = y[i];}
    for (int i = 0, j = K4 + model_size; i < K4; i++, j++) {xwrap[j] = x[i]; ywrap[j] = y[i];}

    double *wx = new double[model_size + 2 * K4];
    // memset(wx, 0, (model_size + 2 * K4));
    for (int i = 0; i < model_size + 2 * K4; i++) wx[i] = 0;
    calw(wx, xwrap, K, K4, H, model_size);

    // Fill the W buffers
    for (int i = model_size, j = 0; i < model_size + K4; i++, j++) wx[j] = wx[i];
    for (int i = K4, j = model_size + K4; i < 2 * K4; i++, j++) wx[j] = wx[i];

    // memset(dz, 0, model_size);
    for (int i = 0; i < model_size; i++) dz[i] = 0;
    caldz(dz, wx, xwrap, ywrap, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_size, model_number);

    delete[] x;
    delete[] y;
    delete[] xwrap;
    delete[] ywrap;
    delete[] wx;
}


void step_L04_ens(double *zens, double *a, int model_size, int ensemble_size, double smooth_steps, double ss2, double space_time_scale, double sts2, double coupling, double forcing, int K, int K2, int K4, int H, int model_number, double delta_t)
{
    for (int iens = 0; iens < ensemble_size; iens++)
    {
        double *z = new double[model_size];
        for (int i = 0; i < model_size; i++) z[i] = zens[iens * model_size + i];
        double *z_save = new double[model_size];
        // memcpy(z_save, z, model_size);
        for (int i = 0; i < model_size; i++) z_save[i] = z[i];

        double *dz = new double[model_size];
        double *z1 = new double[model_size];
        double *z2 = new double[model_size];
        double *z3 = new double[model_size];
        double *z4 = new double[model_size];

        comp_dt_L04(dz, z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number);
        for (int i = 0; i < model_size; i++)
        {
            z1[i] = delta_t * dz[i];
            z[i] = z_save[i] + z1[i] / 2.0;
        }

        comp_dt_L04(dz, z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number);
        for (int i = 0; i < model_size; i++)
        {
            z2[i] = delta_t * dz[i];
            z[i] = z_save[i] + z2[i] / 2.0;
        }

        comp_dt_L04(dz, z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number);
        for (int i = 0; i < model_size; i++)
        {
            z3[i] = delta_t * dz[i];
            z[i] = z_save[i] + z3[i];
        }

        comp_dt_L04(dz, z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number);
        for (int i = 0; i < model_size; i++)
        {
            z4[i] = delta_t * dz[i];
            // z[i] = z_save[i] + (z1[i] + 2.0 * z2[i] + 2.0 * z3[i] + z4[i]) / 6.0;
            z[i] = z_save[i] + z1[i] / 6.0 + z2[i] / 3.0 + z3[i] / 3.0 + z4[i] / 6.0;
        }

        for (int i = 0; i < model_size; i++) zens[iens * model_size + i] = z[i];

        delete[] z;
        delete[] z_save;
        delete[] dz;
        delete[] z1;
        delete[] z2;
        delete[] z3;
        delete[] z4;
    }
}

void step_L04_single(double *z, double *a, int model_size, double smooth_steps, double ss2, double space_time_scale, double sts2, double coupling, double forcing, int K, int K2, int K4, int H, int model_number, double delta_t)
{
    double *z_save = new double[model_size];
    // memcpy(z_save, z, model_size);
    for (int i = 0; i < model_size; i++) z_save[i] = z[i];

    double *dz = new double[model_size];
    double *z1 = new double[model_size];
    double *z2 = new double[model_size];
    double *z3 = new double[model_size];
    double *z4 = new double[model_size];

    comp_dt_L04(dz, z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number);
    for (int i = 0; i < model_size; i++)
    {
        z1[i] = delta_t * dz[i];
        z[i] = z_save[i] + z1[i] / 2.0;
    }

    comp_dt_L04(dz, z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number);
    for (int i = 0; i < model_size; i++)
    {
        z2[i] = delta_t * dz[i];
        z[i] = z_save[i] + z2[i] / 2.0;
    }

    comp_dt_L04(dz, z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number);
    for (int i = 0; i < model_size; i++)
    {
        z3[i] = delta_t * dz[i];
        z[i] = z_save[i] + z3[i];
    }

    comp_dt_L04(dz, z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number);
    for (int i = 0; i < model_size; i++)
    {
        z4[i] = delta_t * dz[i];
        // z[i] = z_save[i] + (z1[i] + 2.0 * z2[i] + 2.0 * z3[i] + z4[i]) / 6.0;
        z[i] = z_save[i] + z1[i] / 6.0 + z2[i] / 3.0 + z3[i] / 3.0 + z4[i] / 6.0;
    }

    delete[] z_save;
    delete[] dz;
    delete[] z1;
    delete[] z2;
    delete[] z3;
    delete[] z4;

}


class LoadBalancer
{
    public:
        LoadBalancer(int num_threads, int ensemble_size, int model_size, double* zens)
        {
            this->num_threads = num_threads;
            this->ensemble_size = ensemble_size;
            this->model_size = model_size;
            this->zens = zens;
        }

        void run(double *a, int smooth_steps, double ss2, double space_time_scale, double sts2, double coupling, double forcing, int K, int K2, int K4, int H, int model_number, double delta_t)
        {
            std::thread producerThread(&LoadBalancer::producer, this);
            std::thread consumerThreads[num_threads];

            for (int i = 0; i < num_threads; ++i) 
            {
                consumerThreads[i] = std::thread(&LoadBalancer::consumer, this, a, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number, delta_t);
            }

            producerThread.join();
            for (int i = 0; i < num_threads; ++i) 
            {
                consumerThreads[i].join();
            }
        }

    private:
        std::queue<double*> buffer;
        std::mutex mtx;
        std::condition_variable producer_cv, consumer_cv;

        int num_threads;
        int ensemble_size;
        int model_size;
        double* zens;
        
        void producer()
        {
            for (int iens = 0; iens < ensemble_size; iens++)
            {
                std::unique_lock<std::mutex> lock(mtx);
                producer_cv.wait(lock, [this] { return buffer.size() < static_cast<unsigned>(ensemble_size); });

                double *z = zens + iens * model_size;
                buffer.push(z);

                consumer_cv.notify_one();
            }
        }

        void consumer(double *a, int smooth_steps, double ss2, double space_time_scale, double sts2, double coupling, double forcing, int K, int K2, int K4, int H, int model_number, double delta_t)
        {
            while(true)
            {
                std::unique_lock<std::mutex> lock(mtx);
                consumer_cv.wait(lock, [this] { return !buffer.empty() || buffer.size() == static_cast<unsigned>(ensemble_size); });

                if (buffer.empty() && buffer.size() == static_cast<unsigned>(ensemble_size)) 
                {
                    // All resources consumed and produced, exit the consumer
                    return;
                }

                double *z = buffer.front();
                buffer.pop();

                step_L04_single(z, a, model_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number, delta_t);

                producer_cv.notify_one();
            }
        }
};


py::array_t<double> step_L04(py::array_t<double> zens_np, py::array_t<double> a_np, int model_size, int ensemble_size, double smooth_steps, double ss2, double space_time_scale, double sts2, double coupling, double forcing, int K, int K2, int K4, int H, int model_number, double delta_t)
{
    py::buffer_info zens_info = zens_np.request();
    py::buffer_info a_info = a_np.request();

    double *zens = (double *)zens_info.ptr;
    double *a = (double *)a_info.ptr;

    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    // std::queue<double*> buffer(num_threads);
    

    int mems_per_thread = ensemble_size / num_threads;
    int extra_mems = ensemble_size % num_threads;

    int start_iens = 0;
    int end_iens = 0;

    for (int i = 0; i < num_threads; i++)
    {
        start_iens = end_iens;
        end_iens = start_iens + mems_per_thread + (i < extra_mems ? 1 : 0);
        threads.emplace_back(step_L04_ens, zens + start_iens * model_size, a, model_size, end_iens - start_iens, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number, delta_t);
    }

    // step_L04_cpp(zens, a, model_size, ensemble_size, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number, delta_t);

    for (auto &thread : threads)
    {
        thread.join();
    }

    // LoadBalancer lb(num_threads, ensemble_size, model_size, zens);
    // lb.run(a, smooth_steps, ss2, space_time_scale, sts2, coupling, forcing, K, K2, K4, H, model_number, delta_t);

    return zens_np;
}


PYBIND11_MODULE(cpu_parallel, m)
{
    m.doc() = "step_L04 cpp extension, cpu version, multi-threaded";
    m.def("step_L04", &step_L04, "step_L04 function");
}