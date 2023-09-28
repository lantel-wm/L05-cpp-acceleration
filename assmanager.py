import os
import json
import numpy as np
import configparser
from scipy.io import loadmat
import ast
import operator as op
from tqdm import tqdm

from filter.EnKF import EnKF
from filter.EnSRF import EnSRF
from model.Lorenz05 import Lorenz05


class AssManager:
    """ Data assimilation manager
    """
    def __init__(self, config = 'config.ini') -> None:
        
        # load config file
        if type(config) not in [str, dict]:
            raise ValueError(f'Invalid config type {type(config)}, must be str or dict')
        
        # load config file
        if type(config) == str:
            if not os.path.exists(config):
                raise ValueError(f'Invalid config path {config}, file not exists')
        
            with open(config, 'r') as f:
                conpar = configparser.ConfigParser()
                conpar.optionxform = lambda option: option
                conpar.read(config)
                
            self.__check_config_file(conpar, config)
            
            self.config = {s:dict(self.__type_recovery(conpar.items(s))) for s in conpar.sections()}
        
        # load config dict
        elif type(config) == dict:
            self.__check_config_dict(config)
            
            with open('config.ini', 'r') as f:
                conpar = configparser.ConfigParser()
                conpar.optionxform = lambda option: option
                conpar.read('config.ini')
            
            # load default config
            self.config = {s:dict(self.__type_recovery(conpar.items(s))) for s in conpar.sections()}
            # overwrite default config
            for section in config:
                for option in config[section]:
                    self.config[section][option] = config[section][option]
        
        self.verbose = self.config['Experiment_option']['verbose']
        if self.verbose:
            self.__show_logo()
            
        # correct time_steps in model and filter
        if self.config['model_params']['time_steps'] != self.config['DA_params']['time_steps']:
            print(f'\n\nWarning: time_steps in model and filter are not equal.\n\n')
            print(f'time_steps in model: {self.config["model_params"]["time_steps"]}')
            print(f'time_steps in filter: {self.config["DA_params"]["time_steps"]}')
            correction_value = min(self.config['model_params']['time_steps'], self.config['DA_params']['time_steps'])
            print(f'\n\ntime_steps in filter will be set to {correction_value}.\n\n')
            self.config['model_params']['time_steps'] = correction_value
            self.config['DA_params']['time_steps'] = correction_value
            print('Continue? (y/n)')
            
            while True:
                choice = input()
                if choice in ['y', 'Y', 'yes', 'Yes', 'YES']:
                    break
                elif choice in ['n', 'N', 'no', 'No', 'NO']:
                    print('Experiment terminated.')
                    exit(0)
            
        
        # load model and filter
        self.model = Lorenz05(self.config['model_params'])
        
        self.filter = self.__select_filter(self.config['DA_config']['filter'])
        
        # load data
        zics_total = loadmat(self.config['Input_file_paths']['ics_path'])[self.config['Input_file_paths']['ics_key']]
        zobs_total = loadmat(self.config['Input_file_paths']['obs_path'])[self.config['Input_file_paths']['obs_key']]
        ztruth_total = loadmat(self.config['Input_file_paths']['truth_path'])[self.config['Input_file_paths']['truth_key']]      
        
        # set intial conditions
        ensemble_size = self.config['DA_config']['ensemble_size']
        ics_imem_beg = self.config['IC_data']['ics_imem_beg'] # initial condition ensemble member id begin
        ics_imem_end = ics_imem_beg + ensemble_size # initial condition ensemble member id end
        self.zens = np.mat(zics_total[ics_imem_beg:ics_imem_end, :]) # ic
        
        # set observations
        time_steps = self.config['model_params']['time_steps']
        obs_freq_step = self.config['DA_params']['obs_freq_timestep']
        iobs_beg = 0
        iobs_end = int(time_steps / obs_freq_step) + 1
        self.zobs_total = np.mat(zobs_total[iobs_beg:iobs_end, :]) # obs
        
        # set truth
        itruth_beg = 0
        itruth_end = int(time_steps / obs_freq_step) + 1
        self.ztruth_total = np.mat(ztruth_total[itruth_beg:itruth_end, :]) # truth
        
        self.zt = self.ztruth_total * self.filter.Hk.T
        self.obs_error_std = np.std(self.zt - self.zobs_total)
        
        # record initial state
        self.initial_state = {
            'zens.shape': str(self.zens.shape),
            'zics_total.shape': str(zics_total.shape),
            'zobs_total.shape': str(self.zobs_total.shape),
            'ztruth_total.shape': str(self.ztruth_total.shape),
            'Hk.shape': str(self.filter.Hk.shape),
            'obs_error_std': self.obs_error_std,
            'total_DA_cycles': self.filter.nobstime,
            'model_grids': str(self.filter.model_grids[:5].T.tolist()) + '...' + str(self.filter.model_grids[-5:].T.tolist()),
            'obs_grids': str(self.filter.obs_grids[:5].T.tolist()) + '...' + str(self.filter.obs_grids[-5:].T.tolist()),
        }
        
    # public methods
    def run(self) -> None:
        """ run the experiment
        """        
        if self.verbose:
            self.__show_run_info()
        
        # DA cycles
        loop = tqdm(range(self.filter.nobstime), desc=self.config['Experiment_option']['experiment_name'])
        for iassim in loop:
            zobs = self.zobs_total[iassim, :]
            z_truth = self.ztruth_total[iassim, :]
            
            if self.filter.inflation_sequence == 'before_DA':
                zens_prior = self.zens
                zens_inf = self.filter.inflation(zens_prior)
                zens_analy = self.filter.assimalate(zens_inf, zobs)
                self.zens = zens_analy
                
            elif self.filter.inflation_sequence == 'after_DA':
                zens_prior = self.zens
                zens_analy = self.filter.assimalate(zens_prior, zobs)
                zens_inf = self.filter.inflation(zens_analy)
                self.zens = zens_inf
            
            # save data
            self.filter.save_current_state(zens_prior, zens_analy, zens_inf, z_truth)
            
            # advance model
            for _ in range(self.filter.obs_freq_timestep):
                self.zens = self.model.step_L04(self.zens)
        
        self.result_save_path = os.path.join(self.config['Experiment_option']['result_save_path'], self.config['Experiment_option']['experiment_name'])
        print(f'\n\nSaving experiment result in {self.result_save_path}...\n\n')
        if not os.path.exists(self.result_save_path):
            os.makedirs(self.result_save_path)
        else:
            print(f'Warning: {self.result_save_path} already exists, incrementing path...\n\n')
            self.result_save_path = increment_path(self.result_save_path)
            print(f'Incremented path to {self.result_save_path}\n\n')
            
        self.filter.save(self.config['Experiment_option'], self.result_save_path,  self.ztruth_total, self.zobs_total)
        self.__save_config()
        
        
    # private methods
    def __check_config_file(self, conpar:configparser.ConfigParser, config:str) -> None:
        """ check the config file

        Args:
            conpar (configparser.ConfigParser): config parser
            config (str): config file path

        Raises:
            ValueError: Invalid section: _section_ in _config_
            ValueError: Invalid option: _option_ in section _section_ of _config_
        """
        template = configparser.ConfigParser()
        template.read('template.ini')
        
        template_sections = template.sections()
        
        for section in conpar.sections():
            if section not in template_sections:
                raise ValueError(f'Invalid section: {section} in {config}')
            
            template_options = template.options(section)
            for option in conpar.options(section):
                if option not in template_options:
                    raise ValueError(f'Invalid option: {option} in section {section} of {config}')
    
    
    def __check_config_dict(self, config:dict) -> None:
        """ check the config dict

        Args:
            config (dict): config dict

        Raises:
            ValueError: Invalid section: _section_ in _config_
            ValueError: Invalid option: _option_ in section _section_ of _config_
        """
        template = configparser.ConfigParser()
        template.read('template.ini')
        
        template_sections = template.sections()
        
        for section in config:
            if section not in template_sections:
                raise ValueError(f'Invalid section: {section}')
            
            template_options = template.options(section)
            for option in config[section]:
                if option not in template_options:
                    raise ValueError(f'Invalid option: {option} in section {section} of config dict')
    
    
    def __is_legal_expr(self, expr:str) -> bool:
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
        

    def __type_recovery(self, items:list) -> list:
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
            elif self.__is_legal_expr(value):
                yield key, eval_expr(value)
            else:
                yield key, value     
    
                
    def __select_filter(self, filter_name:str) -> EnKF:
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
        
    def __show_logo(self) -> None:
        print('''
|---------------------------------------------------------------------------------------|
|    _                          _____ _____                    _   _                    |
|   | |                        |  _  |  ___|                  | | | |                   |
|   | | ___  _ __ ___ _ __  ___| |/' |___ \ ______ _ __  _   _| |_| |__   ___  _ __     |
|   | |/ _ \| '__/ _ \ '_ \|_  /  /| |   \ \______| '_ \| | | | __| '_ \ / _ \| '_ \    |
|   | | (_) | | |  __/ | | |/ /\ |_/ /\__/ /      | |_) | |_| | |_| | | | (_) | | | |   |
|   |_|\___/|_|  \___|_| |_/___|\___/\____/       | .__/ \__, |\__|_| |_|\___/|_| |_|   |
|                                                 | |     __/ |                         |
|                                                 |_|    |___/                          |
|                                                                                       |
|---------------------------------------------------------------------------------------|                                           
              ''')
        
    def __show_run_info(self) -> None:
        
        print('\n\n Experiment settings: \n\n')
        print(json.dumps(self.config['model_params'], indent=4))
        print(json.dumps(self.config['DA_params'], indent=4))
        print(json.dumps(self.config['DA_config'], indent=4))
        print('\n\n Initial state: \n\n')
        print(json.dumps(self.initial_state, indent=4))
        print('\n\n-----------------------------------------------------------------------------------------\n\n')
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
        
    def __save_config(self) -> None:
        """ save config file as .ini
        """
        config = configparser.ConfigParser()
        # self.config is dict
        config.read_dict(self.config)
    
        with open(os.path.join(self.result_save_path, 'config.ini'), 'w') as f:
            config.write(f)
        
        print('\n\n-----------------------------------------------------------------------------------------\n\n')
        print(f'Experiment result saved in {self.result_save_path}.\n')
        print('\n\n👍 Experiment finished.\n\n')


operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
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
    config = {
        'model_params': {
            'forcing': 20.0,
            'time_steps': 50,
        },
        'DA_params': {
            'time_steps': 50,
        }
    }
    am = AssManager(config)
    am.run()
        
        
