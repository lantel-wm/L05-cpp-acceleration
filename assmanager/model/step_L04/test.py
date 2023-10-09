import sys
sys.path.append('..')
import cpu
import gpu
from Lorenz05 import Lorenz05
from Lorenz05 import calx, calw, caldz
import numpy as np
import time


def test_calx():
    params = {}
    model = Lorenz05(params)

    z = np.mat(np.random.randn(model.model_size, 1))
    zwrap = np.concatenate([z[(model.model_size - model.ss2 - 1): model.model_size], z, z[0: model.ss2]])
    x = np.mat(np.zeros((model.model_size, 1)))

    t1 = time.time()
    x1 = calx(x, zwrap, model.a, model.model_size, model.ss2, model.smooth_steps)
    print(time.time() - t1)

    t2 = time.time()
    x2 = cpu.calx(x, zwrap, model.a, model.model_size, model.ss2, model.smooth_steps)
    print(time.time() - t2)

    print((x1 == x2).T)

    
def test_calw():
    params = {}
    model = Lorenz05(params)
    
    wx = np.mat(np.random.randn(model.model_size + model.K4 * 2, 1))
    x = np.mat(np.random.randn(model.model_size, 1))
    xwrap = np.concatenate([x[(model.model_size - model.K4 - 1): model.model_size], x, x[0: model.K4]])
    
    t1 = time.time()
    wx1 = calw(wx, xwrap, model.K, model.K4, model.H, model.model_size)
    print(time.time() - t1)
    
    t2 = time.time()
    wx2 = cpu.calw(wx, xwrap, model.K, model.K4, model.H, model.model_size)
    print(time.time() - t2)
    
    print((wx1 == wx2).T)
    

def test_caldz():
    params = {}
    model = Lorenz05(params)
    
    wx = np.mat(np.random.randn(model.model_size + model.K4 * 2, 1))
    x = np.mat(np.random.randn(model.model_size, 1))
    xwrap = np.concatenate([x[(model.model_size - model.K4 - 1): model.model_size], x, x[0: model.K4]])
    dz = np.mat(np.zeros((model.model_size, 1)))
    y = np.mat(np.random.randn(model.model_size, 1))
    ywrap = np.concatenate([y[(model.model_size - model.K4 - 1): model.model_size], y, y[0: model.K4]])
    
    t1 = time.time()
    dz1 = caldz(wx, xwrap, dz, ywrap, model.space_time_scale, model.sts2, model.coupling, model.forcing, model.K, model.K2, model.K4, model.H, model.model_size, model.model_number)
    print(time.time() - t1)
    
    t2 = time.time()
    dz2 = cpu.caldz(wx, xwrap, dz, ywrap, model.space_time_scale, model.sts2, model.coupling, model.forcing, model.K, model.K2, model.K4, model.H, model.model_size, model.model_number)
    print(time.time() - t2)
    
    print((dz1 == dz2).T)


def test_calx_gpu():
    params = {}
    model = Lorenz05(params)
    
    ensemble_size = 40
    zens = np.mat(np.random.randn(ensemble_size, model.model_size))
    zens_wrap = np.concatenate([zens[:, (model.model_size - model.ss2 - 1): model.model_size], zens, zens[:, 0: model.ss2]], axis=1)
    xens = np.mat(np.zeros((ensemble_size, model.model_size)))
    
    t1 = time.time()

    
if __name__ == '__main__':
    test_calx()
    test_calw()
    test_caldz()