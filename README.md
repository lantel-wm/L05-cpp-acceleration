# lorenz05-cpp-acceleration

[简体中文](./README_zh-CN.md)

[![GitHub License](https://img.shields.io/github/license/ZZY000926/assmanager)](https://github.com/ZZY000926/assmanager/blob/main/LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/ZZY000926/assmanager)](https://github.com/ZZY000926/assmanager/issues)
[![GitHub Stars](https://img.shields.io/github/stars/ZZY000926/assmanager)](https://github.com/ZZY000926/assmanager/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ZZY000926/assmanager)](https://github.com/ZZY000926/assmanager/network)

Python package for data assimilation experiments with the Lorenz 05 model, with cpp acceleration.

## Updates

### 2023.10.11 v1.0.1: Model advance rewritten in C++ with multi-threading.

- The Lorenz 05 model advance is now rewritten in C++ with multi-threading, resulting in a **10x speedup** compared to the original Python code. (Tested on Intel Core i9-13900K, 6400MHz 64G DDR5 RAM)

### 2023.10.8 v1.0.0: Initial release.


## Features

assmanager encapsulates the Lorenz 05 model and data assimilation processes, offering high usability and extensibility.

- Define custom experiment parameters using either ini files or Python dictionaries for easy configuration of multiple experiments.
- Each experiment's results and parameters are saved in a separate folder for easy replication.
- Modular design for easy addition of new data assimilation methods and inflation/localization schemes.
- High speed. The Lorenz 05 model advance is written in C++.

## Installation

Install `assmanager` using the following command:

``` bash
python setup.py install
```

To uninstall `assmanager`, use:

``` bash
pip uninstall assmanager
```

## Quick Start


``` python
from assmanager import AssManager

am = AssManager()
am.run()
```

Running the above code will perform a data assimilation experiment with default parameters for the Lorenz 05 model.

## Usage

First, you need to set experiment parameters. Assmanager supports setting experiment parameters using Python dictionaries or ini configuration files.

### Setting Experiment Parameters

When using a dictionary, the structure passed to AssManager is as follows:

``` python
config = {
    'model_params': {
        'model_size': 960, 
        'forcing': 15.00,
        'time_steps': 200 * 360, 
        ...
    },

    'DA_params': {
        'obs_density': 4,
        'obs_freq_timestep': 50,
        'obs_error_var': 1.0,
    },

    'DA_option': {
        'save_prior_ensemble': False,
        'save_prior_mean': True,
        ...
    },

    'Input_file_paths': {
        'ics_path': 'data/ics_ms3_from_zt1year_sz3001.mat',
        ...
    },

    'IC_data': {
        'ics_imem_beg': 248,
        ...
    },

    'Experiment_option': {
        'verbose': True,
        'experiment_name': 'test',
        ...
    }
}
```

Instantiate AssManager with the config:

``` python
am = AssManager(config)
```


When using an ini configuration file, first edit the ini configuration file and then pass the ini file directory when instantiating AssManager:

``` python
config_path = './config.ini'
am = AssManager(config_path)
```

### Default Experiment Parameters and Meanings

When no custom parameter configuration is provided, assmanager will use the default config.ini file with the following content:

``` ini
[model_params]
model_size = 960                    # N
forcing = 15.00                     # F
space_time_scale = 10.00            # b
coupling = 3.00                     # c
smooth_steps = 12                   # I
K = 32
delta_t = 0.001
time_step_days = 0
time_step_seconds = 432
time_steps = 200 * 360              # 1 day~ 200 time steps
model_number = 3                    # 2 scale

[DA_params]
model_size = 960
time_steps = 200 * 360              # 1 day~ 200 time steps
obs_density = 4
obs_freq_timestep = 50
obs_error_var = 1.0

[DA_config]
ensemble_size = 40
filter = EnKF
update_method = serial_update               # serial_update, parallel_update
inflation_method = multiplicative           # None, multiplicative
inflation_factor = 1.01
inflation_sequence = before_DA              # before_DA, after_DA
localization_method = GC                    # None, GC, CNN
localization_radius = 240

[DA_option]
save_prior_ensemble = False
save_prior_mean = True
save_analysis_ensemble = False
save_analysis_mean = True
save_observation = False
save_truth = False
save_kalman_gain = False
save_prior_rmse = True
save_analysis_rmse = True
save_prior_spread_rmse = True
save_analysis_spread_rmse = True

[Input_file_paths]
ics_path = /data1/zyzhao/scratch/data/ics_ms3_from_zt1year_sz3001.mat # directory must exist
ics_key = zics_total1                                                 # optionnal, if needed
obs_path = /data1/zyzhao/scratch/data/obs_ms3_err1_240s_6h_5y.mat     # directory must exist
obs_key = zobs_total                                                  # optionnal, if needed
truth_path = /data1/zyzhao/scratch/data/zt_25year_ms3_6h.mat          # directory must exist
truth_key = zens_times                                                # optionnal, if needed

[IC_data]
ics_imem_beg = 248

[Experiment_option]
verbose = True
experiment_name = default
result_save_path = /data1/zyzhao/L05_experiments                      # must be valid path
data_save_path = data                       # data folder name, choose what u like
file_save_type = npy                        # npy, mat, dat
prior_ensemble_filename = zens_prior        # prior ensemble file will be saved as zens_prior.npy
prior_mean_filename = prior                 # the same as above
analysis_ensemble_filename = zens_analy
analysis_mean_filename = analy
obs_filename = zobs
truth_filename = ztruth
kalman_gain_filename = kg
prior_rmse_filename = prior_rmse
analysis_rmse_filename = analy_rmse
prior_spread_rmse_filename = prior_spread_rmse
analysis_spread_rmse_filename = analy_spread_rmse
```

The parameter configuration is divided into seven major sections:

- **model_params**: Configuration of parameters for the Lorenz 05 model.
- **DA_params**: External parameter settings used for data assimilation, including the total number of integration steps, observation density, observation frequency, and more.
- **DA_config**: Internal parameter settings for data assimilation, including ensemble size, data assimilation algorithm, inflation, and localization settings.
- **DA_option**: Data saving switches to specify which data to save, including ensemble mean, Kalman gain, RMSE, and more.
- **Input_file_paths**: Configuration for reading external data, such as initial conditions, observation data, and ground truth data file directories.
- **IC_data**: Configuration for specifying which ensemble members are included in the initial conditions.
- **Experiment_option**: Experiment name, experiment save directory, data save format, and file name settings.

### Running Multiple Experiments

``` python
from assmanager import AssManager

inflation_values = [1.0, 1.03, 1.05]
inflation_sequences = ['before_DA', 'after_DA']

configs = []

for inf in inflation_values:
    for seq in inflation_sequences:
        configs.append(
            {
                'DA_config': {
                    'inflation_factor': inf,
                    'inflation_sequence': seq,
                },

                'Experiment_option': {
                    'experiment_name': f'inf_{inf}_{seq}'
                }
            }
        )

ams = [AssManager(config) for config in configs]
for am in ams:
    am.run()
```

## Extensions

This framework supports the extension of data assimilation (DA) algorithms, inflation methods, and localization methods.

### inflation

To add a new inflation method, open `filter/ensembleFilter.py`, find the `inflation` function within the `ensembleFilter` class, and add an additional `if` statement for your new method:

``` python
# filter/ensembleFilter.py

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
    
    elif self.inflation_method == 'your inflation method':
        # some calculations here
        # if you need previous prior and analysis mean, you can access them by self.prior_mean and self.analy_mean
        zens_inf = ...
        return zens_inf
```

### localization

To add a new localization method, open `filter/ensembleFilter.py`, find the `construct_GC_2d function`, create a new function to compute the localization matrix ρ, and modify the `__get_localization_matrix` method in the `ensembleFilter` class to use your new localization method.

### DA method

To add a new DA method, copy the `EnKF.py` file in the filter/ directory and rename it to `your_DA_method.py`. Modify the class name to `yourDAMethod`, and then make changes to the `__serial_update` and `__parallel_update` methods.


## Author

- Zhiyu Zhao
- zyzh@smail.nju.edu.cn


## Acknowledgments
[Zhongrui Wang](https://github.com/zhongruiw) provided the L05 Python code with Numba acceleration and the DA program.
