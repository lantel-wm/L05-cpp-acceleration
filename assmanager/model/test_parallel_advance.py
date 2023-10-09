from Lorenz05 import Lorenz05
from Lorenz05_gpu import Lorenz05_gpu
import numpy as np
import time

def test_parallel_advance():
    model = Lorenz05({})
    model_gpu = Lorenz05_gpu({})
    
    ensemble_size = 40
    
    # np.random.seed(0)
    zens = np.mat(np.random.randn(ensemble_size, model.model_size))
    zens_gpu = zens.copy()
    
    t1 = time.time()
    zens1 = model.step_L04(zens)
    print(time.time() - t1)
    
    t2 = time.time()
    zens2 = model_gpu.step_L04(zens_gpu)
    print(time.time() - t2)
    
    print((zens1 == zens2).T)
    print(zens1.shape)
    print(zens2.shape)
    print(zens1[0, 0], zens2[0, 0])
    
    
if __name__ == '__main__':
    test_parallel_advance()