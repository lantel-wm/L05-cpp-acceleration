import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# default_config = {
#     "model_params": {
#         "advancement": "cpu_parallel", # default, cpu, cpu_parrellel, unet
#         "model_size": 960,
#         "forcing": 15.0,
#         "space_time_scale": 10.0,
#         "coupling": 3.0,
#         "smooth_steps": 12,
#         "K": 32,
#         "delta_t": 0.001,
#         "time_step_days": 0,
#         "time_step_seconds": 432,
#         "model_number": 3
#     },
#     "DA_params": {
#         "model_size": 960,
#         "time_steps": 200 * 360,
#         "obs_density": 4,
#         "obs_freq_timestep": 50,
#         "obs_error_var": 1.0
#     },
#     "DA_config": {
#         "ensemble_size": 40,
#         "filter": "EnKF",
#         "update_method": "serial_update",
#         "inflation_method": "multiplicative",
#         "inflation_factor": 1.01,
#         "inflation_sequence": "before_DA",
#         "localization_method": "GC",
#         "localization_radius": 240,
#         "cnn_weight_path": ".",
#         "cnn_model_path": ".",
#         "cnn_model_name": "name",
#     },
#     "DA_option": {
#         "save_prior_ensemble": False,
#         "save_prior_mean": True,
#         "save_analysis_ensemble": False,
#         "save_analysis_mean": True,
#         "save_observation": True,
#         "save_truth": False,
#         "save_kalman_gain": False,
#         "save_inflated_kalman_gain": False,
#         "save_localized_kalman_gain": False,
#         "save_inflated_localized_kalman_gain": False,
#         "save_prior_rmse": True,
#         "save_analysis_rmse": True,
#         "save_prior_spread_rmse": True,
#         "save_analysis_spread_rmse": True,
#         "file_save_option": "single_file",
#     },
#     "Input_file_paths": {
#         "ics_path": "/data1/zrwang/data/ics_ms3_from_zt1year_sz3001.mat",
#         "ics_key": "zics_total1",
#         "obs_path": "/data1/zrwang/data/obs_ms3_err1_240s_6h_5y.mat",
#         "obs_key": "zobs_total",
#         "truth_path": "/data1/zrwang/data/zt_5year_ms3_6h.mat",
#         "truth_key": "zens_times"
#     },
#     "IC_data": {
#         "ics_imem_beg": 1
#     },
#     "Experiment_option": {
#         "verbose": True,
#         "experiment_name": "inf_1.0_before_DA",
#         "result_save_path": "/data1/zyzhao/L05_experiments",
#         "data_save_path": "data",
#         "file_save_type": "npy",
#         "prior_ensemble_filename": "zens_prior",
#         "prior_mean_filename": "prior",
#         "analysis_ensemble_filename": "zens_analy",
#         "analysis_mean_filename": "analy",
#         "obs_filename": "zobs",
#         "truth_filename": "ztruth",
#         "kalman_gain_filename": "kg",
#         "inflated_kalman_gain_filename": "kg_inf",
#         "localized_kalman_gain_filename": "kg_loc",
#         "inflated_localized_kalman_gain_filename": "kg_inf_loc",
#         "prior_rmse_filename": "prior_rmse",
#         "analysis_rmse_filename": "analy_rmse",
#         "prior_spread_rmse_filename": "prior_spread_rmse",
#         "analysis_spread_rmse_filename": "analy_spread_rmse"
#     },

class ResultReader:
    def __init__(self, exp_path: str) -> None:
        self.exp_path = exp_path
        self.config
        self.data_path = self._get_data_path()
        
    
    def getvar(self, var_name: str, idxes: range) -> np.ndarray:
        return self._load_data(var_name, idxes)    
    
    
    def _get_data_path(self) -> str:
        """ get data path

        Returns:
            str: data path
        """
        experiment_name = self.config['Experiment_option']['experiment_name']
        result_save_path = self.config['Experiment_option']['result_save_path']
        file_save_option = self.config['DA_option']['file_save_option']
        
        data_path = os.path.join(result_save_path, experiment_name, 'data') if file_save_option == 'single_file' \
            else os.path.join(result_save_path, experiment_name)
            
        return data_path
    
    
    def _load_data(self, data_type: str, idxes: range) -> np.ndarray:
        """ load data from file

        Args:
            data_type (str): data type, one of ['prior_rmse', 'analy_rmse', 'kalman_gain']
            idxes (range): index of data, e.g. range(1000)

        Returns:
            np.ndarray: data
            
        Raises:
            ValueError: invalid data type
        """
        file_save_option = self.config['DA_option']['file_save_option']
        if file_save_option == 'single_file':
            data = np.load(os.path.join(self.data_path, f'{data_type}.npy'))[idxes]
        elif file_save_option == 'multiple_files':
            data_path = os.path.join(self.data_path, data_type)
            if not os.path.exists(data_path):
                raise ValueError(f'Invalid data type: {data_type}')
            
            data = np.zeros((len(idxes), *np.load(os.path.join(data_path, f'{idxes[0]}.npy')).shape))
            for i in idxes:
                data[i] = np.load(os.path.join(data_path, f'{i}.npy'))
            
        return data
        
    