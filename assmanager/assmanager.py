import ast
import time
import configparser
import json
import operator as op
import os

import numpy as np
from .filter import EnKF
from .filter import EnSRF
from .model import Lorenz05, Lorenz05_cpu_parallel, Lorenz05_unet
from scipy.io import loadmat
from tqdm import tqdm
from functools import partial
from copy import deepcopy


class AssManager:
    """ Data assimilation manager
    """
    
    default_config = {
        "model_params": {
            "advancement": "cpu_parallel", # default, cpu, cpu_parrellel, unet
            "model_size": 960,
            "forcing": 15.0,
            "space_time_scale": 10.0,
            "coupling": 3.0,
            "smooth_steps": 12,
            "K": 32,
            "delta_t": 0.001,
            "time_step_days": 0,
            "time_step_seconds": 432,
            "model_number": 3
        },
        "DA_params": {
            "model_size": 960,
            "time_steps": 200 * 360,
            "obs_density": 4,
            "obs_freq_timestep": 50,
            "obs_error_var": 1.0
        },
        "DA_config": {
            "ensemble_size": 40,
            "filter": "EnKF",
            "update_method": "serial_update",
            "inflation_method": "multiplicative",
            "inflation_factor": 1.01,
            "inflation_sequence": "before_DA",
            "localization_method": "GC",
            "localization_radius": 240,
            "cnn_weight_path": ".",
            "cnn_model_path": ".",
            "cnn_model_name": "name",
        },
        "DA_option": {
            "save_prior_ensemble": False,
            "save_prior_mean": True,
            "save_analysis_ensemble": False,
            "save_analysis_mean": True,
            "save_observation": True,
            "save_truth": False,
            "save_kalman_gain": False,
            "save_inflated_kalman_gain": False,
            "save_localized_kalman_gain": False,
            "save_inflated_localized_kalman_gain": False,
            "save_prior_rmse": True,
            "save_analysis_rmse": True,
            "save_prior_spread_rmse": True,
            "save_analysis_spread_rmse": True,
            "file_save_option": "single_file",
        },
        "Input_file_paths": {
            "ics_path": "/data1/zrwang/data/ics_ms3_from_zt1year_sz3001.mat",
            "ics_key": "zics_total1",
            "obs_path": "/data1/zrwang/data/obs_ms3_err1_240s_6h_5y.mat",
            "obs_key": "zobs_total",
            "truth_path": "/data1/zrwang/data/zt_5year_ms3_6h.mat",
            "truth_key": "zens_times"
        },
        "IC_data": {
            "ics_imem_beg": 1
        },
        "Experiment_option": {
            "verbose": True,
            "experiment_name": "inf_1.0_before_DA",
            "result_save_path": "/data1/zyzhao/L05_experiments",
            "data_save_path": "data",
            "file_save_type": "npy",
            "prior_ensemble_filename": "zens_prior",
            "prior_mean_filename": "prior",
            "analysis_ensemble_filename": "zens_analy",
            "analysis_mean_filename": "analy",
            "obs_filename": "zobs",
            "truth_filename": "ztruth",
            "kalman_gain_filename": "kg",
            "inflated_kalman_gain_filename": "kg_inf",
            "localized_kalman_gain_filename": "kg_loc",
            "inflated_localized_kalman_gain_filename": "kg_inf_loc",
            "prior_rmse_filename": "prior_rmse",
            "analysis_rmse_filename": "analy_rmse",
            "prior_spread_rmse_filename": "prior_spread_rmse",
            "analysis_spread_rmse_filename": "analy_spread_rmse"
        },
    }
    
    
    def __init__(self, config = 'config.ini') -> None:
        # load config file
        self.config = self._load_config(config)
        
        self.verbose = self.config['Experiment_option']['verbose']
        if self.verbose:
            self._show_logo()
        
        # load model and filter
        advancement = self.config['model_params']['advancement']
        self.model = self._load_model(advancement)
        self.filter = self._load_filter(self.config['DA_config']['filter'])
        
        # load data and set initial condition, observations and truth
        zics_total, zobs_total, ztruth_total = self._load_data()
        self._set_ic(zics_total)        
        self._set_obs(zobs_total)
        self._set_truth(ztruth_total)
        
        # record initial state
        self._set_initial_state()
        
        # load file save option
        self.file_save_option = self.config['DA_option']['file_save_option']
        if self.file_save_option not in ['single_file', 'multiple_files']:
            raise ValueError(f'Invalid file save option "{self.file_save_option}", must be "single_file" or "multiple_files"')
        
        
    # public methods
    def run(self) -> None:
        """ run the experiment
        """        
        if self.verbose:
            self._show_run_info()
            
        # num_process = self.config['DA_config']['ensemble_size']
        # num_process = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(num_process)
        
        if self.file_save_option == 'multiple_files':
            self.result_save_path = os.path.join(self.config['Experiment_option']['result_save_path'], self.config['Experiment_option']['experiment_name'])
            print(f'\n\nSaving experiment result in {self.result_save_path}...\n\n')
            if not os.path.exists(self.result_save_path):
                os.makedirs(self.result_save_path)
            else:
                print(f'Warning: {self.result_save_path} already exists, incrementing path...\n\n')
                self.result_save_path = increment_path(self.result_save_path)
                print(f'Incremented path to {self.result_save_path}\n\n')
            self._save_config()
    
        # DA cycles
        loop = tqdm(range(self.filter.nobstime), desc=self.config['Experiment_option']['experiment_name'])
        for iassim in loop:
            zobs = self.zobs_total[iassim, :]
            z_truth = self.ztruth_total[iassim, :]
            
            # t1 = time.time()
            if self.filter.inflation_sequence == 'before_DA':
                zens_prior = self.zens
                zens_inf = self.filter.inflate(zens_prior)
                zens_analy = self.filter.assimalate(zens_inf, zobs)
                self.zens = zens_analy
                
            elif self.filter.inflation_sequence == 'after_DA':
                zens_prior = self.zens
                zens_analy = self.filter.assimalate(zens_prior, zobs)
                zens_inf = self.filter.inflate(zens_analy)
                self.zens = zens_inf
            # print(f'assimalate time: {time.time() - t1}')
            
            # save data
            if self.file_save_option == 'single_file':
                self.filter.save_current_state(zens_prior, zens_analy, z_truth)
                
            elif self.file_save_option == 'multiple_files':
                self.filter.save_current_state_file(zens_prior, zens_analy, z_truth, zobs, self.result_save_path)
            
            # advance model
            # parallel_step_forward(pool, num_process, self.zens, self.filter.obs_freq_timestep, self.model)
            # t1 = time.time()
            if self.advancement == 'unet':
                self.zens = self.model.step_L04(self.zens)
            else:
                for _ in range(self.filter.obs_freq_timestep):
                    self.zens = self.model.step_L04(self.zens)
            # print(f'advance model time: {time.time() - t1}')
        
        if self.file_save_option == 'single_file':
            self.result_save_path = os.path.join(self.config['Experiment_option']['result_save_path'], self.config['Experiment_option']['experiment_name'])
            print(f'\n\nSaving experiment result in {self.result_save_path}...\n\n')
            if not os.path.exists(self.result_save_path):
                os.makedirs(self.result_save_path)
            else:
                print(f'Warning: {self.result_save_path} already exists, incrementing path...\n\n')
                self.result_save_path = increment_path(self.result_save_path)
                print(f'Incremented path to {self.result_save_path}\n\n')
            
            self.filter.save(self.config['Experiment_option'], self.result_save_path,  self.ztruth_total, self.zobs_total)
            self._save_config()
            
        self._done()
        
        
    # private methods
    def _check_config_file(self, conpar:configparser.ConfigParser, config:str) -> None:
        """ check the config file

        Args:
            conpar (configparser.ConfigParser): config parser
            config (str): config file path

        Raises:
            ValueError: Invalid section: _section_ in _config_
            ValueError: Invalid option: _option_ in section _section_ of _config_
        """
        # template = configparser.ConfigParser()
        # template.read('template.ini')
        
        template_sections = self.default_config.keys()
        
        for section in self.config.keys():
            if section not in template_sections:
                raise ValueError(f'Invalid section: {section} in {config}')
            
            template_options = self.default_config[section].keys()
            for option in conpar.options(section):
                if option not in template_options:
                    raise ValueError(f'Invalid option: {option} in section {section} of {config}')
    
    
    def _check_config_dict(self, config:dict) -> None:
        """ check the config dict

        Args:
            config (dict): config dict

        Raises:
            ValueError: Invalid section: _section_ in _config_
            ValueError: Invalid option: _option_ in section _section_ of _config_
        """
        # template = configparser.ConfigParser()
        # template.read('template.ini')
        
        template_sections = self.default_config.keys()
        
        for section in config:
            if section not in template_sections:
                raise ValueError(f'Invalid section: {section}')
            
            template_options = self.default_config[section].keys()
            for option in config[section]:
                if option not in template_options:
                    raise ValueError(f'Invalid option: {option} in section {section} of config dict')
    
    
    def _is_legal_expr(self, expr:str) -> bool:
        """ check if the expression is legal

        Args:
            expr (str): expression

        Returns:
            bool: True if legal, False otherwise
        """
        try:
            eval_expr(expr)
            return True
        except:
            return False
        

    def _type_recovery(self, items:list) -> list:
        """ recover the type of the value from str to the original type

        Args:
            items (list): list of (key, value) tuples, value is str

        Yields:
            Iterator[list]: iterator of (key, value) tuples, value is original type
        """
        for key, value in items:
            
            # strip the comment '#' in the value
            value = value.split('#')[0].strip()
    
            if value in ['True', 'False']:
                yield key, bool(value)
            elif value.isdigit():
                yield key, int(value)
            elif value.replace('.', '').isdigit():
                yield key, float(value)
            elif self._is_legal_expr(value):
                yield key, eval_expr(value)
            else:
                yield key, value
                
                
    def _load_config(self, config):
        """ load config

        Args:
            config (str or dict): config file path or config dict

        Raises:
            ValueError: Invalid config type
            ValueError: Invalid config path
        """
        if type(config) not in [str, dict]:
            raise ValueError(f'Invalid config type {type(config)}, must be str or dict')
        if type(config) == str:
            if not os.path.exists(config):
                raise ValueError(f'Invalid config path {config}, file not exists')
        
            with open(config, 'r') as f:
                conpar = configparser.ConfigParser()
                conpar.optionxform = lambda option: option
                conpar.read(config)
                
            self._check_config_file(conpar, config)
            
            self.config = {s:dict(self._type_recovery(conpar.items(s))) for s in conpar.sections()}
        
        # load config dict
        elif type(config) == dict:
            self._check_config_dict(config)
            
            # with open('config.ini', 'r') as f:
            #     conpar = configparser.ConfigParser()
            #     conpar.optionxform = lambda option: option
            #     conpar.read('config.ini')
            
            # load default config
            # self.config = {s:dict(self.__type_recovery(conpar.items(s))) for s in conpar.sections()}
            self.config = deepcopy(self.default_config)
            # overwrite default config
            for section in config:
                for option in config[section]:
                    self.config[section][option] = config[section][option]
                    
                    
    def _load_model(self, advancement: str):
        """ load model

        Args:
            advancement (str): model advancement

        Raises:
            ValueError: Invalid model advancement
        """
        if advancement == 'cpu_parallel':
            return Lorenz05_cpu_parallel(self.config['model_params'])
        elif advancement == 'unet':
            return Lorenz05_unet(self.config['model_params'])
        elif advancement == 'default':
            return Lorenz05(self.config['model_params'])
        else:
            raise ValueError(f'Invalid model advancement {self.advancement}')
    
                
    def _load_filter(self, filter_name:str):
        """ select the filter

        Args:
            filter_name (str): filter name

        Raises:
            ValueError: Invalid filter name

        Returns:
            EnKF: filter
        """
        if filter_name == 'EnKF':
            return EnKF(self.config['DA_params'], self.config['DA_config'], self.config['DA_option'])
        elif filter_name == 'EnSRF':
            return EnSRF(self.config['DA_params'], self.config['DA_config'], self.config['DA_option'])
        else:
            raise ValueError(f'Invalid filter name {filter_name}')
    
    
    def _load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ load data

        Returns:
            tuple: (zics_total, zobs_total, ztruth_total)
        """
        ics_path = self.config['Input_file_paths']['ics_path']
        obs_path = self.config['Input_file_paths']['obs_path']
        truth_path = self.config['Input_file_paths']['truth_path']
        if ics_path.endswith('.mat') and obs_path.endswith('.mat') and truth_path.endswith('.mat'):
            zics_total = loadmat(self.config['Input_file_paths']['ics_path'])[self.config['Input_file_paths']['ics_key']]
            zobs_total = loadmat(self.config['Input_file_paths']['obs_path'])[self.config['Input_file_paths']['obs_key']]
            ztruth_total = loadmat(self.config['Input_file_paths']['truth_path'])[self.config['Input_file_paths']['truth_key']]
        elif ics_path.endswith('.npy') and obs_path.endswith('.npy') and truth_path.endswith('.npy'):
            zics_total = np.load(self.config['Input_file_paths']['ics_path'])
            zobs_total = np.load(self.config['Input_file_paths']['obs_path'])
            ztruth_total = np.load(self.config['Input_file_paths']['truth_path'])
        else:
            raise ValueError(f'Invalid data file type, must be .mat or .npy')
        
        return zics_total, zobs_total, ztruth_total
    
    
    def _set_ic(self, zics_total:np.ndarray):
        """ set initial condition

        Args:
            zics_total (np.ndarray): initial condition

        Returns:
            np.mat: initial condition
        """
        self.zics_total = zics_total
        ensemble_size = self.config['DA_config']['ensemble_size']
        ics_imem_beg = self.config['IC_data']['ics_imem_beg'] # initial condition ensemble member id begin
        ics_imem_end = ics_imem_beg + ensemble_size # initial condition ensemble member id end
        self.zens = np.mat(zics_total[ics_imem_beg:ics_imem_end, :]) # ic
    
    
    def _set_obs(self, zobs_total:np.ndarray):
        """ set observations

        Args:
            zobs_total (np.ndarray): observations

        Returns:
            np.ndarray: observations
        """
        time_steps = self.config['DA_params']['time_steps']
        obs_freq_timestep = self.config['DA_params']['obs_freq_timestep']
        # iobs_beg = int(23 * 360 * 200 / obs_freq_timestep)
        iobs_beg = 0
        iobs_end = iobs_beg + int(time_steps / obs_freq_timestep) + 1
        self.zobs_total = np.mat(zobs_total[iobs_beg:iobs_end, :]) # obs
        if self.zobs_total.shape[1] == self.model.model_size:
            self.zobs_total = self.zobs_total * self.filter.Hk.T
    
    
    def _set_truth(self, ztruth_total:np.ndarray):
        """ set truth

        Args:
            ztruth_total (np.ndarray): truth

        Returns:
            np.ndarray: truth
        """
        time_steps = self.config['DA_params']['time_steps']
        obs_freq_timestep = self.config['DA_params']['obs_freq_timestep']
        # itruth_beg = int(23 * 360 * 200 / obs_freq_timestep)
        itruth_beg = 0
        itruth_end = itruth_beg + int(time_steps / obs_freq_timestep) + 1
        self.ztruth_total = np.mat(ztruth_total[itruth_beg:itruth_end, :]) # truth
        
        self.zt = self.ztruth_total * self.filter.Hk.T
        self.obs_error_std = np.std(self.zt - self.zobs_total)
        
        
    def _set_initial_state(self):
        """ set initial state
        """
        self.initial_state = {
            'zens.shape': str(self.zens.shape),
            'zics_total.shape': str(self.zics_total.shape),
            'zobs_total.shape': str(self.zobs_total.shape),
            'ztruth_total.shape': str(self.ztruth_total.shape),
            'Hk.shape': str(self.filter.Hk.shape),
            'obs_error_std': self.obs_error_std,
            'total_DA_cycles': self.filter.nobstime,
            'model_grids': str(self.filter.model_grids[:5].T.tolist()) + '...' + str(self.filter.model_grids[-5:].T.tolist()),
            'obs_grids': str(self.filter.obs_grids[:5].T.tolist()) + '...' + str(self.filter.obs_grids[-5:].T.tolist()),
        }
        
    
    def _show_logo(self) -> None:
        print('''
              
|-------------------------------------------------------------|
|                                                             |
|     _            __  __                                     |
|    / \   ___ ___|  \/  | __ _ _ __   __ _  __ _  ___ _ __   |
|   / _ \ / __/ __| |\/| |/ _` | '_ \ / _` |/ _` |/ _ \ '__|  |
|  / ___ \\\__ \__ \ |  | | (_| | | | | (_| | (_| |  __/ |     |
| /_/   \_\___/___/_|  |_|\__,_|_| |_|\__,_|\__, |\___|_|     |
|                                           |___/             |
|                                                             |
|-------------------------------------------------------------|

              ''')
        
    def _show_run_info(self) -> None:
        
        print('\n\n Experiment settings: \n\n')
        print(json.dumps(self.config['model_params'], indent=4))
        print(json.dumps(self.config['DA_params'], indent=4))
        print(json.dumps(self.config['DA_config'], indent=4))
        print('\n\n Initial state: \n\n')
        print(json.dumps(self.initial_state, indent=4))
        print('\n\n-------------------------------------------------------------\n\n')
        # while True:
        #     print('Start experiment with the above configurations? (y/n)')
        #     choice = input()
        #     if choice in ['y', 'Y', 'yes', 'Yes', 'YES']:
        #         print('\n\n🚀 Experiment started.\n\n')
        #         break
        #     elif choice == ['n', 'N', 'no', 'No', 'NO']:
        #         print('Experiment terminated.')
        #         exit(0)
        
        print('\n\n🚀 Experiment started.\n\n')
        
    def _save_config(self) -> None:
        """ save config file as .ini
        """
        config = configparser.ConfigParser()
        # self.config is dict
        config.read_dict(self.config)
    
        with open(os.path.join(self.result_save_path, 'config.ini'), 'w') as f:
            config.write(f)

    def _done(self):
        print('\n\n-------------------------------------------------------------n\n')
        print(f'Experiment result saved in {self.result_save_path}.\n')
        print('\n\n👍 Experiment finished.\n\n')

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}


def eval_(node):
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)


def eval_expr(expr: str):
    """ evaluate the expression

    Args:
        expr (str): expression

    Returns:
        
    """
    return eval_(ast.parse(expr, mode='eval').body)


def increment_path(fpath: str):
    """ increment the path

    Args:
        fpath (str): file path

    Returns:
        str: incremented file path
    """
    path_idx = 2
    if fpath.endswith('/'):
        fpath = fpath[:-1]
        
    while True:
        inc_fpath = f'{fpath}_{path_idx}'
        # print(f'Incrementing path to {inc_fpath}')
        if not os.path.exists(inc_fpath):
            os.makedirs(inc_fpath)
            return inc_fpath
        else:
            path_idx += 1
     
                
if __name__ == '__main__':
    pass
        
