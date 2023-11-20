import numpy as np
from numba import jit, njit, prange
from .step_L04 import cpu_parallel


class Lorenz05_cpu_parallel:
    """ Lorenz05 model

    Raises:
        ValueError: Invalid parameter
            in __init__(), if the parameter is not in parameter_list
        ValueError: Invalid model number
            in __comp_dt_L04(), if the model number is not 2 or 3
    """
    # default parameters
    # model parameters
    model_size = 960                    # N
    forcing = 15.00                     # F
    space_time_scale = 10.00            # b
    coupling = 3.00                     # c
    smooth_steps = 12                   # I
    K = 32
    delta_t = 0.001
    time_step_days = 0
    time_step_seconds = 432
    model_number = 3                    # 2 scale
    
    parameter_list = [
        'advancement',
        'model_size',
        'forcing',
        'space_time_scale',
        'coupling',
        'smooth_steps', 
        'K',
        'delta_t',
        'time_step_days',
        'time_step_seconds',
        'model_number',
    ]
    
    
    def __init__(self, params:dict) -> None:
        """ model parameters initialization

        Args:
            parameters (dict): parameters of the model

        Raises:
            ValueError: invalid parameter
        """
        for key in params:
            if key not in self.parameter_list:
                raise ValueError(f'Invalid parameter: {key}')
            setattr(self, key, params[key])
            
        # for speed
        self.H = int(self.K / 2)
        self.K2 = 2 * self.K
        self.K4 = 4 * self.K
        self.ss2 = 2 * self.smooth_steps
        self.sts2 = self.space_time_scale ** 2

        # smoothing filter
        self.alpha = (3.0 * (self.smooth_steps ** 2) + 3.0) / (2.0 * (self.smooth_steps ** 3) + 4.0 * self.smooth_steps)
        self.beta = (2.0 * (self.smooth_steps ** 2) + 1.0) / (1.0 * (self.smooth_steps ** 4) + 2.0 * (self.smooth_steps ** 2))

        ri = - self.smooth_steps - 1.0
        j = 0
        self.a = np.zeros(2*self.smooth_steps+1)
        for i in range(-self.smooth_steps, self.smooth_steps+1):
            ri = ri + 1.0
            self.a[j] = self.alpha - self.beta * abs(ri)
            j = j + 1

        self.a[0] = self.a[0] / 2.00
        self.a[2 * self.smooth_steps] = self.a[2 * self.smooth_steps] / 2.00
        
        # model advance step counter (istep)
        self.advance_step_counter = 0
    
        
    # public method   
    def step_L04(self, zens:np.mat) -> np.mat: 
        """ integrate the model for one time step

        Args:
            z (np.mat): ensemble state of the model
            
        Returns:
            np.mat: state of the model after integration
        """
        
        zens = cpu_parallel.step_L04(zens, self.a, self.model_size, zens.shape[0], self.smooth_steps, self.ss2, self.space_time_scale, self.sts2, self.coupling, self.forcing, self.K, self.K2, self.K4, self.H, self.model_number, self.delta_t)
            
        # update model advance step counter
        self.advance_step_counter += 1
        
        return zens