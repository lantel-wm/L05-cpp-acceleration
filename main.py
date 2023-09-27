import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from filter.EnKF import EnKF
from model.Lorenz05 import Lorenz05

time_steps = 200 * 2  # 360 days (1 day~ 200 time steps)

config_dict = {
    'model_params': {
        'forcing': 16.0,
        'time_steps': time_steps,
        'model_number': 3,
    },
    
    'DA_params': {
        'time_steps': time_steps,
        'obs_density': 4,
        'obs_freq_timestep': 50,
        'obs_error_var': 1.0,
    },
    
    'DA_config': {
        'ensemble_size': 40,
        'filter' : 'EnKF',
        'update_method': 'serial_update',
        'inflation_method': 'multiplicative',
        'inflation_factor': 1.01,
        'inflation_sequence': 'before DA',
        'localization_method': 'GC',
        'localization_radius': 240,
    },
    
    'DA_option': {
        'save_prior_ensemble': False,
        'save_prior_mean': True,
        'save_analysis_ensemble': False,
        'save_analysis_mean': True,
        'save_observation': False,
        'save_truth': False,
        'save_kalman_gain': False,
        'save_prior_rmse': True,
        'save_analysis_rmse': True,
        'save_prior_spread_rmse': True,
        'save_analysis_spread_rmse': True,
    },
    
    'Input_file_paths': {
        'ics_path': '/data1/zyzhao/scratch/data/ics_ms3_from_zt1year_sz3001.mat',
        'ics_key': 'zics_total1',
        'obs_path': '/data1/zyzhao/scratch/data/obs_ms3_err1_240s_6h_5y.mat',
        'obs_key': 'zobs_total',
        'truth_path': '/data1/zyzhao/scratch/data/zt_25year_ms3_6h.mat',
        'truth_key': 'zens_times',
    },
    
    'Experiment_option': {
        'verbose': True,
        'experiment_name': 'default',
        
        'result_save_path': '/data1/zyzhao/L05_experiments',
        'data_save_path': 'data',
        
        'file_save_type': 'npy',
        'prior_ensemble_filename': 'zens_prior', # zens_prior.npy
        'prior_mean_filename': 'prior', # zens_prior_mean.npy
        'analysis_ensemble_filename': 'zens_analy', # zens_analy.npy
        'analysis_mean_filename': 'analy', # zens_analy_mean.npy
        'obs_filename': 'zobs', # zobs.npy
        'truth_filename': 'ztruth', # zt.npy
        'kalman_gain_filename': 'kg', # kg.npy
        'prior_rmse_filename': 'prior_rmse', # prior_rmse.npy
        'analysis_rmse_filename': 'analy_rmse', # analy_rmse.npy
        'prior_spread_rmse_filename': 'prior_spread_rmse', # prior_spread_rmse.npy
        'analysis_spread_rmse_filename': 'analy_spread_rmse', # analy_spread_rmse.npy
    }
}


time_steps = config_dict['model_params']['time_steps']
obs_freq_step = config_dict['DA_params']['obs_freq_timestep']
ensemble_size = config_dict['DA_config']['ensemble_size']
obs_density = config_dict['DA_params']['obs_density']

zics_total = loadmat(config_dict['Input_file_paths']['ics_path'])[config_dict['Input_file_paths']['ics_key']]
zobs_total = loadmat(config_dict['Input_file_paths']['obs_path'])[config_dict['Input_file_paths']['obs_key']]
ztruth_total = loadmat(config_dict['Input_file_paths']['truth_path'])[config_dict['Input_file_paths']['truth_key']]

ics_imem_beg = 248 # initial condition ensemble member id begin
ics_imem_end = ics_imem_beg + ensemble_size # initial condition ensemble member id end
zens = np.mat(zics_total[ics_imem_beg:ics_imem_end, :]) # ic

iobs_beg = 0
iobs_end = int(time_steps / obs_freq_step) + 1
zobs_total = np.mat(zobs_total[iobs_beg:iobs_end, :]) # obs

itruth_beg = 0 # truth state id begin
itruth_end = int(time_steps / obs_freq_step) + 1 # truth state id end
ztruth_total = np.mat(ztruth_total[itruth_beg:itruth_end, :]) # truth

model = Lorenz05(config_dict['model_params'])
filter = EnKF(config_dict['DA_params'], config_dict['DA_config'], config_dict['DA_option'])

zt = ztruth_total * filter.Hk.T

print('truth shape:', np.shape(zt))
print('observations error std:', np.std(zt - zobs_total))

for iassim in tqdm(range(filter.nobstime)):
    zobs = zobs_total[iassim, :]
    z_truth = ztruth_total[iassim, :]
    
    if filter.inflation_sequence == 'before DA':
        zens_prior = zens
        zens_inf = filter.inflation(zens_prior)
        zens_analy = filter.assimalate(zens_inf, zobs)
        zens = zens_analy
        
    elif filter.inflation_sequence == 'after DA':
        zens_prior = zens
        zens_analy = filter.assimalate(zens_prior, zobs)
        zens_inf = filter.inflation(zens_analy)
        zens = zens_inf
    
    # save data
    filter.save_current_state(zens_prior, zens_analy, zens_inf, z_truth)
    
    # advance model
    for _ in range(obs_freq_step):
        zens = model.step_L04(zens)
        
filter.save(config_dict['Experiment_option'], ztruth_total)