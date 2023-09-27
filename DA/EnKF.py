import numpy as np
from numba import jit
from ensembleFilter import ensembleFilter

class EnKF(ensembleFilter):
    def __init__(self, params: dict, config: dict, options: dict) -> None:
        super().__init__(params, config, options)
        
        if self.localization_type == 'GC':
            self.CMat = self.__construct_GC_2d(self.localization_value)
    
    # public methods
    def assimalate(self, zens: np.mat, obs: np.mat) -> np.mat:
        if self.update_method == 'serial_update':
            return self.__serial_update(zens, obs)
        
    # private methods
    @jit(nopython=True)
    def construct_GC_2d(self, cut, ylocs):
        l = self.model_size
        ylocs = self.obs_grids
        nobs = len(ylocs)
        V = np.zeros((nobs, l))

        for iobs in range(0, nobs):
            yloc = ylocs[iobs]
            for iCut in range(0, l):
                dist = min(abs(iCut+1 - yloc), abs(iCut+1 - l - yloc), abs(iCut+1 + l - yloc))
                r = dist / (0.5 * cut)

                if dist >= cut:
                    V[iobs, iCut] = 0.0
                elif 0.5*cut <= dist < cut:
                    V[iobs, iCut] = r**5 / 12.0 - r**4 / 2.0 + r**3 * 5.0 / 8.0 + r**2 * 5.0 / 3.0 - 5.0 * r + 4.0 - 2.0 / (3.0 * r)
                else:
                    V[iobs, iCut] = r**5 * (-0.25) + r**4 / 2.0 + r**3 * 5.0/8.0 - r**2 * 5.0/3.0 + 1.0

        return V
    
    def __serial_update(self, zens, obs):
        rn = 1.0 / (self.ensemble_size - 1)

        for iobs in range(self.nobsgrid):
            xmean = np.mean(zens, axis=0)  # 1xn
            xprime = zens - xmean
            hxens = (self.Hk[iobs, :] * zens.T).T  # 40*1
            hxmean = np.mean(hxens, axis=0)
            hxprime = hxens - hxmean
            hpbht = (hxprime.T * hxprime * rn)[0, 0]
            pbht = (xprime.T * hxprime) * rn
        
            if self.localization_type == 'GC':
                Cvect = CMat[iobs, :]
                kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
            else:
                kfgain = pbht / (hpbht + obs_error_var)

            inc = (kfgain * (zobs[:,iobs] - hxens).T).T

            zens = zens + inc

        return zens