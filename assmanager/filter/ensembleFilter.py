# -*- coding: utf-8 -*-

# Copyright © 2023 Zhongrui Wang & Zhiyu Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Coded by: Zhongrui Wang & Zhiyu Zhao

import os
from abc import ABC, abstractmethod
import numpy as np
from numba import jit


class ensembleFilter(ABC):
    """ ensemble filter abstract base class

    Inherit:
        ABC: abstract base class

    Raises:
        ValueError: Invalid parameter
        ValueError: Invalid configuration
        ValueError: Invalid option
    """
    
    # filter parameters
    model_size = 960
    time_steps = 200 * 360              # 360 days(1 day~ 200 time steps)
    obs_density = 4
    obs_freq_timestep = 50
    obs_error_var = 1.0
    
    # filter configurations
    ensemble_size = 960
    filter = 'EnKF'                     # 'EnKF', 'EnSRF'
    update_method = 'serial_update'     # 'serial_update', 'parallel_update'
    inflation_method = 'multiplicative' # None, 'multiplicative', 'RTPS', 'RTPP', 'CNN'
    inflation_factor = 1.01             # None, float
    inflation_sequence = 'after_DA'     # 'before_DA', 'after_DA'
    localization_method = 'GC'          # None, 'GC', 'CNN'
    localization_radius = 240           # None, float
    
    # running options
    save_prior_ensemble = False         # save prior ensemble
    save_prior_mean = True              # save prior mean
    save_analysis_ensemble = False      # save analysis ensemble
    save_analysis_mean = True           # save analysis mean
    save_observation = False            # save observations
    save_truth = False                  # save ground truth
    save_kalman_gain = False            # save kalman gain matrix
    save_prior_rmse = True              # save prior rmse with time
    save_analysis_rmse = True           # save analysis rmse with time
    save_prior_spread_rmse = True       # save prior spread rmse with time
    save_analysis_spread_rmse = True    # save analysis spread rmse with time
    
    parameter_list = [
        'model_size',
        'time_steps',
        'obs_density',
        'obs_freq_timestep',
        'obs_error_var',
    ]
    
    configuration_list = [
        'ensemble_size',
        'filter',
        'update_method',
        'inflation_method',
        'inflation_factor',
        'inflation_sequence',
        'localization_method',
        'localization_radius',
    ]
    
    option_list = [
        'save_prior_ensemble',
        'save_prior_mean',
        'save_analysis_ensemble',
        'save_analysis_mean',
        'save_observation',
        'save_truth',
        'save_kalman_gain',
        'save_prior_rmse',
        'save_analysis_rmse',
        'save_prior_spread_rmse',
        'save_analysis_spread_rmse',
    ]
    
    def __init__(self, params:dict, config:dict, options:dict) -> None:
        """ ensemble filter parameters initialization

        Args:
            params (dict): filter parameters
            config (dict): filter configurations
            options (dict): running options

        Raises:
            ValueError: Invalid parameter
            ValueError: Invalid configuration
            ValueError: Invalid option
        """
        for key in params:
            if key not in self.parameter_list:
                raise ValueError(f'Invalid parameter: {key}')
            setattr(self, key, params[key])    
        
        for key in config:
            if key not in self.configuration_list:
                raise ValueError(f'Invalid configuration: {key}')
            setattr(self, key, config[key])
            
        for key in options:
            if key not in self.option_list:
                raise ValueError(f'Invalid option: {key}')
            setattr(self, key, options[key])
            
        # observation parameters
        self.model_grids = np.arange(1, self.model_size + 1)
        self.obs_grids = self.model_grids[self.model_grids % self.obs_density == 0]
        self.nobsgrid = len(self.obs_grids)
        
        self.model_times = np.arange(0, self.time_steps + 1)
        self.obs_times = self.model_times[self.model_times % self.obs_freq_timestep == 0]
        self.nobstime = len(self.obs_times)
        
        self.R = np.mat(np.eye(self.nobsgrid) * self.obs_error_var)
        
        # forward operator
        self.Hk = np.mat(np.zeros((self.nobsgrid, self.model_size)))
        for iobs in range(self.nobsgrid):
            x1 = self.obs_grids[iobs] - 1
            self.Hk[iobs, x1] = 1.0
        
        # GC localization
        if self.localization_method == 'GC':
            self.CMat = np.mat(self.__get_localization_matrix())
            
        # array for saving results
        self.zens_prior = np.zeros((self.nobstime, self.ensemble_size, self.model_size)) if self.save_prior_ensemble else None
        
        self.prior = np.zeros((self.nobstime, self.model_size))
        
        self.zens_analy = np.zeros((self.nobstime, self.ensemble_size, self.model_size)) if self.save_analysis_ensemble else None
        
        self.analy = np.zeros((self.nobstime, self.model_size))
            
        self.kg_series = np.zeros((self.nobstime, self.model_size, self.nobsgrid)) if self.save_kalman_gain else None
        
        self.prior_rmse = np.zeros(self.nobstime) if self.save_prior_rmse else None
        
        self.analy_rmse = np.zeros(self.nobstime) if self.save_analysis_rmse else None
            
        self.prior_spread = np.zeros((self.nobstime, self.model_size))
        
        self.analy_spread = np.zeros((self.nobstime, self.model_size))
        
        self.prior_spread_rmse = np.zeros(self.nobstime) if self.save_prior_spread_rmse else None
            
        self.analy_spread_rmse = np.zeros(self.nobstime) if self.save_analysis_spread_rmse else None
            
        # assimilation step counter (iassim)
        self.assimilation_step_counter = 0


    # public methods    
    def inflation(self, zens: np.mat) -> np.mat:
        """ inflation

        Args:
            zens (np.mat): state ensemble
            zens_prior (np.mat): prior state ensemble

        Returns:
            np.mat: inflated state ensemble
        """
        if self.inflation_method is None:
            return zens
        
        elif self.inflation_method == 'multiplicative':
            ens_mean = np.mean(zens, axis=0)
            ens_prime = zens - ens_mean
            zens_inf = ens_mean + self.inflation_factor * ens_prime
            return zens_inf
        
        elif self.inflation_method == 'RTPS':
            # TODO: RTPS inflation
            pass
            # std_prior = np.std(zens_prior, axis=0, ddof=1)
            # std_analy = np.std(zens, axis=0, ddof=1)
            # ens_mean = np.mean(zens, axis=0)
            # ens_prime = zens - ens_mean
            # zens_inf = ens_mean + np.multiply(ens_prime, (1 + self.inflation_factor * (std_prior - std_analy) / std_analy))
            # return zens_inf
            
    
    def save(self, save_config:dict, result_save_path: str, ztruth_total:np.mat, zobs_total:np.mat) -> None:
        data_save_path = save_config['data_save_path']
        data_save_path = os.path.join(result_save_path, data_save_path)
        
        print(data_save_path)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
            
        file_save_type = save_config['file_save_type']
        if file_save_type == 'npy':
            if self.save_prior_ensemble:
                prior_filename = save_config['prior_ensemble_filename'] + '.' + file_save_type
                prior_save_path = os.path.join(data_save_path, prior_filename)
                np.save(prior_save_path, self.zens_prior)
                
            if self.save_prior_mean:
                prior_mean_filename = save_config['prior_mean_filename'] + '.' + file_save_type
                prior_mean_save_path = os.path.join(data_save_path, prior_mean_filename)
                np.save(prior_mean_save_path, self.prior)
                
                
            if self.save_analysis_ensemble:
                analysis_filename = save_config['analysis_ensemble_filename'] + '.' + file_save_type
                analysis_save_path = os.path.join(data_save_path, analysis_filename)
                np.save(analysis_save_path, self.zens_analy)
                
            if self.save_analysis_mean:
                analysis_mean_filename = save_config['analysis_mean_filename'] + '.' + file_save_type
                analysis_mean_save_path = os.path.join(data_save_path, analysis_mean_filename)
                np.save(analysis_mean_save_path, self.analy)
                
            if self.save_observation:
                obs_filename = save_config['obs_filename'] + '.' + file_save_type
                obs_save_path = os.path.join(data_save_path, obs_filename)
                np.save(obs_save_path, zobs_total)
                
            if self.save_truth:
                truth_filename = save_config['truth_filename'] + '.' + file_save_type
                truth_save_path = os.path.join(data_save_path, truth_filename)
                np.save(truth_save_path, ztruth_total)
                
            if self.save_kalman_gain:
                kg_filename = save_config['kalman_gain_filename'] + '.' + file_save_type
                kg_save_path = os.path.join(data_save_path, kg_filename)
                np.save(kg_save_path, self.kg_series)
                
            if self.save_prior_rmse:
                self.prior_rmse = np.sqrt(np.mean(np.square(self.prior - ztruth_total), axis=1))
                prior_rmse_filename = save_config['prior_rmse_filename'] + '.' + file_save_type
                prior_rmse_save_path = os.path.join(data_save_path, prior_rmse_filename)
                np.save(prior_rmse_save_path, self.prior_rmse)
                
            if self.save_analysis_rmse:
                self.analy_rmse = np.sqrt(np.mean(np.square(self.analy - ztruth_total), axis=1))
                analysis_rmse_filename = save_config['analysis_rmse_filename'] + '.' + file_save_type
                analysis_rmse_save_path = os.path.join(data_save_path, analysis_rmse_filename)
                np.save(analysis_rmse_save_path, self.analy_rmse)
                
            if self.save_prior_spread_rmse:
                self.prior_spread_rmse = np.sqrt(np.mean(np.square(self.prior_spread), axis=1))
                prior_spread_rmse_filename = save_config['prior_spread_rmse_filename'] + '.' + file_save_type
                prior_spread_rmse_save_path = os.path.join(data_save_path, prior_spread_rmse_filename)
                np.save(prior_spread_rmse_save_path, self.prior_spread_rmse)
                
            if self.save_analysis_spread_rmse:
                self.analy_spread_rmse = np.sqrt(np.mean(np.square(self.analy_spread), axis=1))
                analysis_spread_rmse_filename = save_config['analysis_spread_rmse_filename'] + '.' + file_save_type
                analysis_spread_rmse_save_path = os.path.join(data_save_path, analysis_spread_rmse_filename)
                np.save(analysis_spread_rmse_save_path, self.analy_spread_rmse)
                
        else:
            # TODO: save data in other format
            pass
            
            
    def save_current_state(self, zens_prior:np.mat, zens_analy:np.mat, zens_inf:np.mat, z_truth:np.mat) -> None:
        """ save current state

        Args:
            zens_prior (np.mat): prior state ensemble
            zens_analy (np.mat): posterior state ensemble
            zens_inf (np.mat): posterior state ensemble after inflation
            zens_truth (np.mat): true state
        """
        if self.save_prior_ensemble:
            self.zens_prior[self.assimilation_step_counter - 1, :, :] = zens_prior
        
        if self.save_analysis_ensemble:
            self.zens_analy[self.assimilation_step_counter - 1, :, :] = zens_analy
            
        if self.save_kalman_gain:
            self.kg_series[self.assimilation_step_counter - 1, :, :] = self.calc_current_kalman_gain_matrix(zens_inf)

        self.analy[self.assimilation_step_counter - 1, :] = np.mean(zens_analy, axis=0)
        self.prior[self.assimilation_step_counter - 1, :] = np.mean(zens_prior, axis=0)
        
        self.prior_spread[self.assimilation_step_counter - 1, :] = np.std(zens_prior, axis=0, ddof=1)
        self.analy_spread[self.assimilation_step_counter - 1, :] = np.std(zens_analy, axis=0, ddof=1)
        
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
    
    
    def calc_prior_rmse(self, zens_prior: np.mat, z_truth: np.mat) -> float:
        """ calculate the prior rmse

        Args:
            zens_prior (np.mat): prior state ensemble
            zt (np.mat): ground truth

        Returns:
            float: current prior rmse
        """
        prior_rmse = np.sqrt(np.mean(np.square(np.mean(zens_prior, axis=0) - z_truth)))
        return prior_rmse
    
    
    def calc_analysis_rmse(self, zens_analy: np.mat, z_truth: np.mat) -> float:
        """ calculate the analysis rmse

        Args:
            zens_analy (np.mat): posterior state ensemble
            zt (np.mat): ground truth

        Returns:
            float: current analysis rmse
        """
        analy_rmse = np.sqrt(np.mean(np.square(np.mean(zens_analy, axis=0) - z_truth)))
        return analy_rmse
    
    
    def calc_prior_spread_rmse(self, zens_prior: np.mat) -> float:
        """ calculate the prior spread rmse

        Args:
            zens_prior (np.mat): prior state ensemble

        Returns:
            float: current prior spread rmse
        """
        prior_spread = np.std(zens_prior, axis=0, ddof=1)
        prior_spread_rmse = np.sqrt(np.mean(np.square(prior_spread)))
        return prior_spread_rmse
    
    
    def calc_analysis_spread_rmse(self, zens_prior: np.mat) -> float:
        """ calculate the analysis spread rmse

        Args:
            zens_prior (np.mat): posterior state ensemble

        Returns:
            float: current analysis spread rmse
        """
        analy_spread = np.std(zens_prior, axis=0, ddof=1)
        analy_spread_rmse = np.sqrt(np.mean(np.square(analy_spread)))
        return analy_spread_rmse
    
    
    # private methods
    def __get_localization_matrix(self) -> np.mat:
        return construct_GC_2d(self.localization_radius, self.model_size, self.obs_grids)
        
    
    # abstract methods
    @abstractmethod
    def assimalate(self, zens:np.mat, obs:np.mat) -> np.mat:
        """ assimalation process

        Args:
            zens (np.mat): prior ensemble state
            obs (np.mat): observations

        Returns:
            np.mat: analysis
        """
        pass


@jit(nopython=True)
def construct_GC_2d(cut: float, l: int, ylocs: np.mat) -> np.mat:
    """ construct the GC localization matrix
    
    Args:
        cut (float): localization radius
        l (int): model size
        ylocs (np.mat): observation locations
    
    Returns:
        np.mat: GC localization matrix
    """

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
