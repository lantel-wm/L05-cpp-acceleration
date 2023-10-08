# assmanager

[![GitHub License](https://img.shields.io/github/license/ZZY000926/assmanager)](https://github.com/ZZY000926/assmanager/blob/main/LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/ZZY000926/assmanager)](https://github.com/ZZY000926/assmanager/issues)
[![GitHub Stars](https://img.shields.io/github/stars/ZZY000926/assmanager)](https://github.com/ZZY000926/assmanager/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ZZY000926/assmanager)](https://github.com/ZZY000926/assmanager/network)

Lorenz 05 model数据同化实验框架的Python包。

## 特性

assmanager对Lorenz 05 model和DA过程进行了封装，具有高易用性和高可扩展性。

- 支持使用ini文件或python字典定义实验参数，方便进行多组实验
- 每次实验结果和实验参数单独保存在一个文件夹，易于复现
- 模块化设计，易于添加新的DA方式和inflation，localization方案
- 对Lorenz 05 model的积分过程进行了numba加速，运行速度快

## 安装

使用以下命令安装 `assmanager`：

``` bash
python setup.py install
```

卸载`assmanager`：
``` bash
pip uninstall assmanager
```

## 快速开始


``` python
from assmanager import AssManager

am = AssManager()
am.run()
```

运行以上代码将进行一次默认参数的Lorenz 05 model的同化实验。

## 用法

首先需要设置实验参数，assmanager支持使用python字典或ini配置文件进行实验参数配置。

### 设置实验参数

使用字典时，传入AssManager的字典结构如下：

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

实例化AssManager时传入config即可：

``` python
am = AssManager(config)
```


使用ini配置文件时，首先需要编辑ini配置文件，然后在实例化AssManager时传入ini文件目录即可。

``` python
config_path = './config.ini'
am = AssManager(config_path)
```

## 默认实验参数及含义

当不传入任何自定义的参数配置时，assmanager将使用默认的`config.ini`文件，内容如下：

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

参数配置分为7个大模块：
- **model_params**: Lorenz 05 model的参数设置
- **DA_params**: 数据同化用到的外部参数设置，包括积分总步数，观测密度，观测频率等
- **DA_config**: 数据同化内部参数设置，包括ensemble size，使用的数据同化算法，inflation和localization设置等
- **DA_option**: 数据保存开关，可以详细设置需要保存哪些数据，包括ensemble mean, kalman gain, rmse等
- **Input_file_paths**: 外部数据读取设置，包括初始场，观测数据，真实数据的文件目录
- **IC_data**: 设置初始场包含哪些ensemble member
- **Experiment_option**: 实验名称，实验保存目录，数据保存格式，文件名设置

## 进行多组实验

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

# 扩展

本框架支持扩展DA算法，inflation算法和localization算法。

## inflation

打开 `filter/ensembleFilter.py`，找到ensembleFilter类中的inflation成员函数，新增一个if即可。

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

## localization

打开 `filter/ensembleFilter.py`，找到construct_GC_2d函数，在下面新增一个计算localization 矩阵 $\rho$ 的函数。

然后修改ensembleFilter的__get_localization_matrix方法，将里面的construct_GC_2d方法改为新添加的localization方法即可。

## DA method

在 `filter/` 目录下将 `EnKF.py` 复制并改名为 `your_DA_method.py` 文件，修改类名为 yourDAMethid，然后只需要修改__serial_update和__parallel_update即可。


# 作者

- Zhiyu Zhao
- zyzh@smail.nju.edu.cn
