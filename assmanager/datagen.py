import os

import numpy as np
from .filter import EnKF
from .filter import EnSRF
from .model import Lorenz05, Lorenz05_gpu, Lorenz05_cpu_parallel
from tqdm import tqdm

class DataGenerator:
    def __init__(self, model_params) -> None:
        self.model = Lorenz05_cpu_parallel(model_params)
        
    def generate_ics(self, ics_size: int, data_save_path='.', random_seed=None):
        """ Generate initial conditions for data assimilation.

        Args:
            ics_size (int): initial conditions size.
            ics_save_path (str, optional): ic file save path. Defaults to '.'.
            random_seed (int, optional): random seed of numpy. Defaults to None.
        """
        time_steps = 200 * 360
        if time_steps // ics_size < 10:
            time_steps = ics_size * 10
        
        years = time_steps // (360 * 200)
        model_size = self.model.model_size
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        z = np.mat(np.random.rand(model_size))
        zics = np.zeros((3001, model_size)) # zics saves last 3001 time steps
        zics_filename = f'ics_ms{self.model.model_number}_from_zt{years}year_sz{ics_size}.npy'
        
        for i in tqdm(range(time_steps), desc='model advancing'):
            z = self.model.step_L04(z)
            if i >= time_steps - ics_size:
                zics[i - time_steps + ics_size] = z
                
        np.random.shuffle(zics)
        np.save(os.path.join(data_save_path, zics_filename), zics)
        
    
    def generate_nr(self, data_save_path='.', time_steps=200 * 360 * 5, obs_freq=50, obs_vars=[1, 2, 4, 8]):
        """ Generate natural run data.

        Args:
            data_save_path (str, optional): data save path. Defaults to '.'.
            time_steps (int, optional): time steps. Defaults to 200 * 360 * 5.
            obs_freq (int, optional): observation frequency. Defaults to 50.
            obs_vars (list, optional): observation variance. Defaults to [1, 2, 4, 8].
        """
        z = np.mat(np.random.rand(self.model.model_size))

        for i in tqdm(range(200 * 360), desc='generate ic'):
            z = self.model.step_L04(z)

        ztruth = np.zeros((time_steps // 50, self.model.model_size))


        for i in tqdm(range(time_steps // 50), desc='generate truth'):
            for _ in range(obs_freq):
                z = self.model.step_L04(z)
            ztruth[i] = z
        
        zobss = []
        for obs_var in obs_vars:
            zobss.append(ztruth + np.random.randn(ztruth.shape[0], ztruth.shape[1]) * np.sqrt(obs_var))

        years = time_steps // (360 * 200)
        obs_freq_hours = int(obs_freq / 200 * 24)
        np.save(os.path.join(data_save_path, f'zt_{years}year_ms{self.model.model_number}_{obs_freq_hours}h.npy'), ztruth)
        for i, obs_var in enumerate(obs_vars):
            np.save(os.path.join(data_save_path, f'obs_ms{self.model.model_number}_var{obs_var}_{self.model.model_size}s_{obs_freq_hours}h_{years}y.npy'), zobss[i])