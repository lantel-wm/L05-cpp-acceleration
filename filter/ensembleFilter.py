from abc import ABC, abstractmethod
import numpy as np
from numba import jit

class ensembleFilter(ABC):
    
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
    inflation_type = 'multiplicative'   # None, 'multiplicative', 'RTPS', 'RTPP', 'CNN'
    inflation_value = 1.01              # None, float
    localization_type = 'GC'            # None, 'GC', 'CNN'
    localization_value = 240            # None, float
    
    # running options
    save_prior = False                  # save prior ensemble
    save_analysis = False               # save analysis ensemble
    save_observation = False            # save observations
    save_kalman_gain = False            # save kalman gain matrix
    save_prior_rmse = False             # save prior rmse with time
    save_analysis_rmse = False          # save analysis rmse with time
    save_prior_spread_rmse = False      # save prior spread rmse with time
    save_analysis_spread_rmse = False   # save analysis spread rmse with time
    result_save_path = './'             # save path
    
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
        'inflation_type',
        'inflation_value',
        'localization_type',
        'localization_value',
    ]
    
    option_list = [
        'save_prior',
        'save_analysis',
        'save_kalman_gain',
        'save_prior_rmse',
        'save_analysis_rmse',
        'save_prior_spread_rmse',
        'save_analysis_spread_rmse',
        'result_save_path'
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
            if key not in self.arguements_list:
                raise ValueError(f'Invalid option: {key}')
            setattr(self, key, options[key])
            
        # observation parameters
        model_grids = np.arange(1, self.model_size + 1)
        obs_grids = model_grids[model_grids % self.obs_density == 0]
        nobsgrid = len(obs_grids)
        
        model_times = np.arange(0, self.time_steps + 1)
        obs_times = model_times[model_times % self.obs_freq_timestep == 0]
        nobstime = len(obs_times)
        
        R = np.mat(np.eye(nobsgrid) * self.obs_error_var)
        
        # forward operator
        Hk = np.mat(np.zeros((nobsgrid, self.model_size)))
        for iobs in range(nobsgrid):
            x1 = obs_grids[iobs] - 1
            Hk[iobs, x1] = 1.0
            
        # array for saving results
        zens_prior = np.zeros((nobstime + 1, self.ensemble_size, self.model_size))
        zens_analy = np.zeros((nobstime + 1, self.ensemble_size, self.model_size))
        kg = np.zeros((nobstime, self.model_size, nobsgrid))
        prior_rmse = np.zeros(nobstime + 1)
        analy_rmse = np.zeros(nobstime + 1)
        prior_spread_rmse = np.zeros(nobstime + 1)
        analy_spread_rmse = np.zeros(nobstime + 1)
    
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
    
    @abstractmethod
    def inflation(self, zens:np.mat) -> np.mat:
        """ inflation process

        Args:
            zens (np.mat): ensemble state

        Returns:
            np.mat: inflated ensemble state
        """
        pass
    
    @abstractmethod            
    def calc_current_kalman_gain_matrix(self, zens_inf:np.mat) -> np.mat:
        """ calculate the current kalman gain matrix

        Args:
            zens (np.mat): ensemble state

        Returns:
            np.mat: kalman gain matrix
        """
        pass
    
    @abstractmethod
    def calc_prior_rmse(self, zens_prior:np.mat, zt:np.mat) -> float:
        """ calculate the prior rmse

        Args:
            zens (np.mat): ensemble state

        Returns:
            float: prior rmse
        """
        pass
    
    @abstractmethod
    def calc_analysis_rmse(self, zens_analy:np.mat, zt:np.mat) -> float:
        """ calculate the analysis rmse

        Args:
            zens (np.mat): ensemble state
            zt (np.mat): ground truth

        Returns:
            float: analysis rmse
        """
        pass
    
    @abstractmethod
    def calc_prior_spread_rmse(self, zens_analy:np.mat) -> float:
        """ calculate the prior spread rmse

        Args:
            zens (np.mat): ensemble state
            zt (np.mat): ground truth

        Returns:
            float: prior spread rmse
        """
        pass
    
    @abstractmethod
    def calc_analysis_spread_rmse(self, zens_prior:np.mat) -> float:
        """ calculate the analysis spread rmse

        Args:
            zens (np.mat): ensemble state

        Returns:
            float: analysis spread rmse
        """
        pass