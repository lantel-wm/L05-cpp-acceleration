import numpy as np
from .ensembleFilter import ensembleFilter

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
            return self.__serial_update(zens, zobs)
        
        elif self.update_method == 'parallel_update':
            # update assimilation step counter
            self.assimilation_step_counter += 1
            return self.__parallel_update(zens, zobs)
        
        
    def inflation(self, zens: np.mat) -> np.mat:
        return super().inflation(zens)
        
            
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
        """ EnKF serial update

        Args:
            zens (np.mat): state ensemble
            zobs (np.mat): observations

        Returns:
            np.mat: analysis
        """
        rn = 1.0 / (self.ensemble_size - 1)
        
        # perturb observations
        obs_p = np.random.normal(0., np.sqrt(self.obs_error_var), (self.ensemble_size, self.nobsgrid))
        obs = zobs + obs_p

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
            else:
                # TODO: other localization methods
                kfgain = pbht / (hpbht + self.obs_error_var)

            inc = (kfgain * (obs[:,iobs] - hxens).T).T

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
        # TODO: parallel update
        pass