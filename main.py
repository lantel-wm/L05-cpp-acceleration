import numpy as np

from .filter import EnKF, EnSRF
from .model import Lorenz05

time_steps = 200 * 360  # 360 days (1 day~ 200 time steps)

config = {
    
    'model_params': {
        'forcing': 15.0,
        'time_steps': time_steps,
        'model_number': 3,
    },
    
    'DA_params': {
        'time_steps': time_steps,
        'obs_density': 4,
        'obs_time_freq_timestep': 50,
        'obs_error_var': 1.0,
    },
    
    'DA_config': {
        'ensemble_size': 40,
        'filter' : 'EnKF',
        'update_method': 'serial_update',
        'inflation_method': 'multiplicative',
        'inflation_factor': 1.01,
        'localization_method': 'GC',
        'localization_radius': 240,
    },
    
    'DA_option': {
        'save_prior': False,
        'save_analysis': False,
        'save_observation': False,
        'save_truth': False,
        'save_kalman_gain': False,
        'save_prior_rmse': True,
        'save_analysis_rmse': True,
        'save_prior_spread_rmse': True,
        'save_analysis_spread_rmse': True,
    },
    
    'Experiment_option': {
        'verbose': True,
        'result_save_path': './',
        'experiment_name': 'default',
    }
}

model = Lorenz05(config['model_params'])
filter = EnKF(config['DA_params'], config['DA_config'], config['DA_option'])

