import numpy as np
from .ensembleFilter import ensembleFilter
# from .acc import cpu
from numba import jit
import torch

class EnKF(ensembleFilter):
    """ Ensemble Kalman Filter

    Inherit:
        ensembleFilter (ABC): ensemble filter abstract base class
    """
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
            # update assimilation step counter
            self.assimilation_step_counter += 1
            
            # perturb observations
            obs_p = np.random.normal(0., np.sqrt(self.obs_error_var), (self.ensemble_size, self.nobsgrid))
            obs = zobs + obs_p
            
            return self.__serial_update(zens, obs)
        
        elif self.update_method == 'parallel_update':
            # update assimilation step counter
            self.assimilation_step_counter += 1
            
            # perturb observations
            obs_p = np.random.normal(0., np.sqrt(self.obs_error_var), (self.ensemble_size, self.nobsgrid))
            obs = zobs + obs_p
            
            return self.__parallel_update(zens, obs)
        
        
    def inflate(self, zens: np.mat) -> np.mat:
        return super().inflate(zens)
    
    
    def save_current_state(self, zens_prior: np.mat, zens_analy: np.mat, z_truth: np.mat) -> None:
        return super().save_current_state(zens_prior, zens_analy, z_truth)
    
    def save_current_state_file(self, zens_prior: np.mat, zens_analy: np.mat, z_truth: np.mat, zobs: np.mat, data_save_path: str) -> None:
        return super().save_current_state_file(zens_prior, zens_analy, z_truth, zobs, data_save_path)
            
    def calc_current_kalman_gain_matrix(self, zens_inf: np.mat, option:str) -> np.mat:
        return super().calc_current_kalman_gain_matrix(zens_inf, option)
    
    
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
        """ EnKF serial update

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
            pbht = (xprime.T * hxprime) * rn
        
            # localization
            if self.localization_method is None:
                kfgain = pbht / (hpbht + self.obs_error_var)
            elif self.localization_method == 'GC':
                Cvect = self.CMat[iobs, :]
                kfgain = np.multiply(Cvect.T, (pbht / (hpbht + self.obs_error_var)))

            inc = (kfgain * (zobs[:,iobs] - hxens).T).T

            zens = zens + inc
        
        return zens
    
    
    def __parallel_update(self, zens:np.mat, zobs:np.mat) -> np.mat:
        """ parallel update

        Args:
            zens (np.mat): state ensemble
            zobs (np.mat): observations

        Returns:
            np.mat: analysis
        """
        rn = 1.0 / (self.ensemble_size - 1)
        Xprime = zens - np.mean(zens, axis=0)
        HXens = (self.Hk * zens.T).T
        HXprime = HXens - np.mean(HXens, axis=0)
        PbHt = (Xprime.T * HXprime) * rn
        HPbHt = (HXprime.T * HXprime) * rn
        K = PbHt * (HPbHt + self.R).I
        
        if self.localization_method == 'GC':
            K = np.multiply(self.CMat.T, K)
        elif self.localization_method == 'CLF':
            K = super()._CLF(K)
        elif self.localization_method is None:
            pass
        
        zens = zens + (K * (zobs - HXens).T).T
        
        return zens
        
    
    
@jit(nopython=True)
def serial_update(zens:np.mat, zobs:np.mat, Hk:np.mat, CMat:np.mat, ensemble_size:int, nobsgrid:int, obs_error_var:float, localization_method:str) -> np.mat:
    rn = 1.0 / (ensemble_size - 1)
    
    for iobs in range(nobsgrid):
        xmean = np.mean(zens, axis=0)  # 1xn
        xprime = zens - xmean
        hxens = (Hk[iobs, :] * zens.T).T  # 40*1
        hxmean = np.mean(hxens, axis=0)
        hxprime = hxens - hxmean
        hpbht = (hxprime.T * hxprime * rn)[0, 0]
        pbht = (xprime.T * hxprime) * rn
    
        # localization
        if localization_method is None:
            kfgain = pbht / (hpbht + obs_error_var)
        elif localization_method == 'GC':
            Cvect = CMat[iobs, :]
            kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
        else:
            # TODO: other localization methods
            kfgain = pbht / (hpbht + obs_error_var)

        inc = (kfgain * (zobs[:,iobs] - hxens).T).T

        zens = zens + inc

    return zens