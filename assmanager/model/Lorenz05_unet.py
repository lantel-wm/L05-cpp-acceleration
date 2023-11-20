import numpy as np
from numba import jit, njit, prange
import torch
from .unet import Unet


class Lorenz05_unet:
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
        
        self.unet = Unet(1, c_expand=32, kernel_sizes=[3, 3, 3, 3, 3], strides=[1, 1, 1, 1, 1])
        checkpoint = torch.load('/home/zzy/zyzhao/Lorenz05/lorenz05-python/assmanager/model/unet/best.pt')
        self.unet.load_state_dict({k.replace('module.',''):v for k, v in checkpoint.items()})
        self.device = torch.device('cuda:0')
        self.unet = self.unet.to(self.device)
    
        
    # public method   
    def step_L04(self, zens:np.mat) -> np.mat: 
        """ integrate the model for one time step

        Args:
            z (np.mat): ensemble state of the model
            
        Returns:
            np.mat: state of the model after integration
        """
        # (40, 960) -> (40, 1, 960)
        zens_blocks = np.array_split(zens, 256, axis=0)
        zens_out = []
        for zens_block in zens_blocks:
            self.unet.eval()
            unet_in = torch.from_numpy(zens_block).float().to(self.device).unsqueeze(1)
            unet_out = self.unet(unet_in).squeeze(1).cpu().detach().numpy()
            zens_out.append(unet_out)
        
        # update model advance step counter
        self.advance_step_counter += 1
        
        zens_out = np.concatenate(zens_out, axis=0)
        # print(np.mean(np.square(zens_out - zens)))
        return zens_out