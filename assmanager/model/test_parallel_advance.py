from Lorenz05 import Lorenz05
# from Lorenz05_gpu import Lorenz05_gpu
from Lorenz05_cpu_parallel import Lorenz05_cpu_parallel
import numpy as np
import time

def test_parallel_advance():
    model = Lorenz05({})
    model_cpu_parallel = Lorenz05_cpu_parallel({})
    
    ensemble_size = 40
    time_steps = 1
    
    np.random.seed(0)
    zens = np.mat(np.random.randn(ensemble_size, model.model_size))
    zens_cpu_parallel = zens.copy()
    
    t1 = time.time()
    for _ in range(time_steps):
        zens = model.step_L04(zens)
    print('time: ', time.time() - t1)
    
    t2 = time.time()
    for _ in range(time_steps):
        zens_cpu_parallel = model_cpu_parallel.step_L04(zens_cpu_parallel)
    print('time: ', time.time() - t2)
    
    print(np.where(zens != zens_cpu_parallel)[0].shape)
    print(np.mean(np.sum((np.square(zens - zens_cpu_parallel)))) / np.where(zens != zens_cpu_parallel)[0].shape[0])
    # print(zens1.shape)
    # print(zens2.shape)
    # print(zens1[0, 0], zens2[0, 0])
    
    
if __name__ == '__main__':
    test_parallel_advance()