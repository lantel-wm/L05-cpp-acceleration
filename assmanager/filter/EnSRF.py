import numpy as np
from .ensembleFilter import ensembleFilter
from numba import jit, njit, prange


class EnSRF(ensembleFilter):
    def __init__(self, params: dict, config: dict, options: dict) -> None:
        super().__init__(params, config, options)
    
    
    # public methods
    def assimalate(self, zens: np.mat, zobs: np.mat) -> np.mat:
        """ assimalate the observations

        Args:
            zens (np.mat): state ensemble
            zobs (np.mat): observations

        Returns:
            np.mat: analysis
        """
        if self.update_method == 'serial_update':
            return self.__serial_update(zens, zobs)
            # return np.mat(serial_update(np.array(zens), np.array(zobs), np.array(self.Hk), np.array(self.CMat), self.ensemble_size, self.nobsgrid, self.obs_error_var, self.localization_method))
        elif self.update_method == 'parallel_update':
            return self.__parallel_update(zens, zobs)
        
        
    def inflate(self, zens: np.mat) -> np.mat:
        return super().inflate(zens)
    
            
    def calc_current_kalman_gain_matrix(self, zens_inf: np.mat) -> np.mat:
        return super().calc_current_kalman_gain_matrix(zens_inf)
    
    
    def calc_prior_rmse(self, zens_prior: np.mat, zt: np.mat) -> float:
        return super().calc_prior_rmse(zens_prior, zt)
    
    
    def calc_analysis_rmse(self, zens_analy: np.mat, zt: np.mat) -> float:
        return super().calc_analysis_rmse(zens_analy, zt)
    
    
    def calc_prior_spread_rmse(self, zens_prior: np.mat) -> float:
        return super().calc_prior_spread_rmse(zens_prior)
    
    
    def calc_analysis_spread_rmse(self, zens_prior: np.mat) -> float:
        return super().calc_analysis_spread_rmse(zens_prior)
    
        
    # private methods
    def __serial_update(self, zens:np.mat, zobs:np.mat) -> np.mat:
        """ EnSRF serial update

        Args:
            zens (np.mat): state ensemble
            zobs (np.mat): observations

        Returns:
            np.mat: analysis
        """
        rn = 1.0 / (self.ensemble_size - 1)
        for iobs in range(self.nobsgrid):
            xmean = np.mean(zens, axis=0)  # 1xn
            xprime = zens - xmean
            hxens = (self.Hk[iobs, :] * zens.T).T  # 40*1
            hxmean = np.mean(hxens, axis=0)
            hxprime = hxens - hxmean
            hpbht = (hxprime.T * hxprime * rn)[0, 0]
            gainfact = (hpbht + self.obs_error_var) / hpbht * (1.0 - np.sqrt(self.obs_error_var / (hpbht + self.obs_error_var)))
            pbht = (xprime.T * hxprime) * rn

            if self.localization_method is None:
                kfgain = pbht / (hpbht + self.obs_error_var)
            elif self.localization_method == 'GC':
                Cvect = self.CMat[iobs, :]
                kfgain = np.multiply(Cvect.T, (pbht / (hpbht + self.obs_error_var)))
            else:
                # TODO: other localization type
                kfgain = pbht / (hpbht + self.obs_error_var)

            mean_inc = (kfgain * (zobs[0, iobs] - hxmean)).T
            prime_inc = - (gainfact * kfgain * hxprime.T).T

            zens = zens + mean_inc + prime_inc

        return zens
    
    
    def __parallel_update(self, zens:np.mat, zobs:np.mat) -> np.mat:
        """ parallel update

        Args:
            zens (np.mat): state ensemble
            zobs (np.mat): observations

        Returns:
            np.mat: analysis
        """
        # TODO: parallel update
        pass

@jit(nopython=True, parallel=True)
def numba_mean(a):
    """ numba mean

    Args:
        a (np.array): array

    Returns:
        np.array: mean
    """
    mean_np = np.zeros((1, a.shape[1]))
    for i in prange(a.shape[1]):
        mean_np[0, i] = a[:, i].mean()
    return mean_np

    
@jit(nopython=True)
def serial_update(zens:np.ndarray, zobs:np.ndarray, Hk:np.ndarray, CMat:np.ndarray, ensemble_size:int, nobsgrid:int, obs_error_var:float, localization_method:str) -> np.ndarray:
    rn = 1.0 / (ensemble_size - 1)
    for iobs in range(nobsgrid):
        # xmean = np.mean(zens, axis=0)  # 1xn
        xmean = numba_mean(zens)

        xprime = zens - xmean
        hxens = (Hk[iobs, :] @ zens.T).T  # 40*1
        # hxmean = np.mean(hxens, axis=0)
        hxmean = hxens.mean()
        hxprime = hxens - hxmean
        hpbht = np.dot(hxprime, hxprime) * rn
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
        pbht = (xprime.T @ hxprime) * rn

        if localization_method is None:
            kfgain = pbht / (hpbht + obs_error_var)
        elif localization_method == 'GC':
            Cvect = CMat[iobs, :]
            kfgain = Cvect.T * (pbht / (hpbht + obs_error_var))
        else:
            # TODO: other localization type
            kfgain = pbht / (hpbht + obs_error_var)

        mean_inc = (kfgain * (zobs[0, iobs] - hxmean))[None, :]
        prime_inc = - (gainfact * hxprime[:, None] @ kfgain[:, None].T)
        # print(zens.shape, mean_inc.shape, prime_inc.shape)

        zens = zens + mean_inc + prime_inc

    return zens