import sys
sys.path.append('..')
import cpu
from EnKF import EnKF
import numpy as np
import time

def test_enkf_serial_update():
    filter = EnKF({}, {}, {})
    
    zens = np.mat(np.random.randn(filter.ensemble_size, filter.model_size))
    zobs = np.mat(np.random.randn(filter.ensemble_size, filter.nobsgrid))
    Hk = filter.Hk
    CMat = filter.CMat
    localize = False
    if filter.localization_method == 'GC':
        localize = True
        
    t1 = time.time()
    zens1 = filter._serial_update(zens, zobs)
    print(time.time() - t1)
    
    t2 = time.time()
    zens2 = cpu.serial_update(zens, zobs, Hk, CMat, filter.nobsgrid, localize, filter.obs_error_var)
    print(time.time() - t2)
    
    print((zens1 == zens2).T)
    print(zens1.shape)
    print(zens2.shape)
    
    
if __name__ == '__main__':
    test_enkf_serial_update()


