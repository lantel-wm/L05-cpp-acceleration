import numpy as np
from numba import jit
from ensembleFilter import ensembleFilter

class EnKF(ensembleFilter):
    def __init__(self, params: dict, config: dict, options: dict) -> None:
        super().__init__(params, config, options)
        
        if self.localization_type == 'GC':
            self.CMat = self.__construct_GC_2d()
    
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
        elif self.update_method == 'parallel_update':
            return self.__parallel_update(zens, zobs)
        
    def inflation(self, zens: np.mat, zens_prior: np.mat) -> np.mat:
        """ inflation

        Args:
            zens (np.mat): state ensemble
            zens_prior (np.mat): prior state ensemble

        Returns:
            np.mat: inflated state ensemble
        """
        if self.inflation_type is None:
            return zens
        elif self.inflation_type == 'multiplicative':
            ens_mean = np.mean(zens, axis=0)
            ens_prime = zens - ens_mean
            zens_inf = ens_mean + self.inflation_value * ens_prime
            return zens_inf
        elif self.inflation_type == 'RTPS':
            std_prior = np.std(zens_prior, axis=0, ddof=1)
            std_analy = np.std(zens, axis=0, ddof=1)
            ens_mean = np.mean(zens, axis=0)
            ens_prime = zens - ens_mean
            zens_inf = ens_mean + np.multiply(ens_prime, (1 + self.inflation_value * (std_prior - std_analy) / std_analy))
            
    def calc_current_kalman_gain_matrix(self, zens_inf: np.mat) -> np.mat:
        """ calculate the current kalman gain matrix

        Args:
            zens_inf (np.mat): inflated state ensemble

        Returns:
            np.mat: current kalman gain matrix
        """
        rn = 1.0 / (self.ensemble_size - 1)
        Xprime = zens_inf - np.mean(zens_inf, axis=0)
        HXens = (self.Hk * zens_inf.T).T
        HXprime = HXens - np.mean(HXens, axis=0)
        PbHt = (Xprime.T * HXprime) * rn
        HPbHt = (HXprime.T * HXprime) * rn
        K = PbHt * (HPbHt + self.R).I
        return K
    
    def calc_prior_rmse(self, zens_prior: np.mat, zt: np.mat) -> float:
        """ calculate the prior rmse

        Args:
            zens_prior (np.mat): prior state ensemble
            zt (np.mat): ground truth

        Returns:
            float: current prior rmse
        """
        prior_rmse = np.sqrt(np.mean((zt - zens_prior) ** 2, axis=0))
        return prior_rmse
    
    def calc_analysis_rmse(self, zens_analy: np.mat, zt: np.mat) -> float:
        """ calculate the analysis rmse

        Args:
            zens_analy (np.mat): posterior state ensemble
            zt (np.mat): ground truth

        Returns:
            float: current analysis rmse
        """
        analy_rmse = np.sqrt(np.mean((zt - zens_analy) ** 2, axis=0))
        return analy_rmse
    
    def calc_prior_spread_rmse(self, zens_prior: np.mat) -> float:
        """ calculate the prior spread rmse

        Args:
            zens_prior (np.mat): prior state ensemble

        Returns:
            float: current prior spread rmse
        """
        prior_spread = np.std(zens_prior, axis=0, ddof=1)
        prior_spread_rmse = np.sqrt(np.mean(prior_spread ** 2, axis=0))
        return prior_spread_rmse
    
    def calc_analysis_spread_rmse(self, zens_prior: np.mat) -> float:
        """ calculate the analysis spread rmse

        Args:
            zens_prior (np.mat): posterior state ensemble

        Returns:
            float: current analysis spread rmse
        """
        analy_spread = np.std(zens_prior, axis=0, ddof=1)
        analy_spread_rmse = np.sqrt(np.mean(analy_spread ** 2, axis=0))
        return analy_spread_rmse
        
        
    # private methods
    @jit(nopython=True)
    def __construct_GC_2d(self) -> np.mat:
        """ construct the GC localization matrix

        Returns:
            np.mat: GC localization matrix
        """
        cut = self.localization_value
        l = self.model_size
        ylocs = self.obs_grids
        nobs = len(ylocs)
        V = np.mat(np.zeros((nobs, l)))

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
            if self.localization_type is None:
                kfgain = pbht / (hpbht + self.obs_error_var)
            elif self.localization_type == 'GC':
                Cvect = self.CMat[iobs, :]
                kfgain = np.multiply(Cvect.T, (pbht / (hpbht + self.obs_error_var)))
            else:
                # TODO: other localization methods
                kfgain = pbht / (hpbht + self.obs_error_var)

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
        # TODO: parallel update
        pass