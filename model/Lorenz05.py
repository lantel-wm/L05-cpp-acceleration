# ----- Lorenz05  -----
# by Zhongrui Wang, Zhiyu Zhao
# version: 1.0
# update: add rtps inf; multiple inf,loc; git repo; allow deflation

import os
import numpy as np
from numba import jit

class Lorenz05:
    """ Lorenz05 model

    Raises:
        ValueError: Invalid parameter
            in __init__(), if the parameter is not in parameter_list
        ValueError: Invalid model number
            in __comp_dt_L04(), if the model number is not 2 or 3
    """
    # default parameters
    # model parameters
    model_size = 960  # N
    forcing = 15.00  # F
    space_time_scale = 10.00  # b
    coupling = 3.00  # c
    smooth_steps = 12  # I
    K = 32
    delta_t = 0.001
    time_step_days = 0
    time_step_seconds = 432
    time_steps = 200 * 360  # 360 days(1 day~ 200 time steps)
    model_number = 3  # 2 scale
    
    parameter_list = [
        'model_size', 'forcing', 'space_time_scale', 'coupling', 'smooth_steps', 
        'K', 'delta_t', 'time_step_days', 'time_step_seconds', 'time_steps', 'model_number'
    ]
    
    # init method
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
        H = int(self.K / 2)
        K2 = 2 * self.K
        K4 = 4 * self.K
        ss2 = 2 * self.smooth_steps
        sts2 = self.space_time_scale ** 2

        # smoothing filter
        alpha = (3.0 * (self.smooth_steps ** 2) + 3.0) / (2.0 * (self.smooth_steps ** 3) + 4.0 * self.smooth_steps)
        beta = (2.0 * (self.smooth_steps ** 2) + 1.0) / (1.0 * (self.smooth_steps ** 4) + 2.0 * (self.smooth_steps ** 2))

        ri = - self.smooth_steps - 1.0
        j = 0
        a = np.zeros(2*self.smooth_steps+1)
        for i in range(-self.smooth_steps, self.smooth_steps+1):
            ri = ri + 1.0
            a[j] = alpha - beta * abs(ri)
            j = j + 1

        a[0] = a[0] / 2.00
        a[2 * self.smooth_steps] = a[2 * self.smooth_steps] / 2.00
            
    # public method    
    def step_L04(self, z:np.mat): 
        """ integrate the model for one time step

        Args:
            z (np.mat): state of the model
        """
        z_save = z
        dz = self.__comp_dt_L04(z)  # Compute the first intermediate step
        z1 = np.multiply(self.delta_t, dz)
        z = z_save + z1 / 2.0

        dz = self.__comp_dt_L04(z)  # Compute the second intermediate step
        z2 = np.multiply(self.delta_t, dz)
        z = z_save + z2 / 2.0

        dz = self.__comp_dt_L04(z)  # Compute the third intermediate step
        z3 = np.multiply(self.delta_t, dz)
        z = z_save + z3

        dz = self.__comp_dt_L04(z)  # Compute fourth intermediate step
        z4 = np.multiply(self.delta_t, dz)

        dzt = z1 / 6.0 + z2 / 3.0 + z3 / 3.0 + z4 / 6.0
        z = z_save + dzt
    
    # private methods
    def __comp_dt_L04(self, z:np.mat) -> np.mat:
        """ compute the time derivative of the model

        Args:
            z (np.mat): state of the model
            
        Raises:
            ValueError: do not know that model number (model_number != 2 or 3)

        Returns:
            np.mat: time derivative of the model
        """
        if self.model_number == 3:
            x, y = self.__z2xy(z)
        elif self.model_number == 2:
            x = z
            y = 0.0 * z
        else:
            # print('Do not know that model number')
            raise ValueError(f'Invalid model number: {self.model_number}, should be 2 or 3')
            # return

        #  Deal with  # cyclic boundary# conditions using buffers

        # Fill the xwrap and ywrap buffers
        xwrap = np.concatenate([x[self.model_size - self.K4: self.model_size], x, x[0: self.K4]])
        ywrap = np.concatenate([y[self.model_size - self.K4: self.model_size], y, y[0: self.K4]])

        wx = np.mat(np.zeros((self.model_size + self.K4 * 2, 1)))
        # ! Calculate the W's
        wx = self.__calw(wx, xwrap)

        # Fill the W buffers
        wx[0: self.K4, 0] = wx[self.model_size: self.model_size + self.K4, 0]
        wx[self.model_size + self.K4: self.model_size + 2 * self.K4, 0] = wx[self.K4: self.K4 * 2, 0]

        dz = np.mat(np.zeros((self.model_size, 1)))
        # ! Generate dz / dt
        dz = self.__caldz(wx, xwrap, dz, ywrap)

        return dz
    
    
    def __z2xy(self, z:np.mat) -> (np.mat, np.mat):
        """ convert z to x and y for model III

        Args:
            z (np.mat): state of the model

        Returns:
            (np.mat, np.mat): x and y
            x: large scale activity
            y: small scale activity
            z = x + y
        """
        # ss2 is smoothing scale I * 2
        # Fill zwrap
        zwrap = np.concatenate([z[(self.model_size - self.ss2 - 1): self.model_size], z, z[0: self.ss2]])
        x = np.mat(np.zeros((self.model_size, 1)))
        y = np.mat(np.zeros((self.model_size, 1)))

        # CPU version -- Generate the x variables
        #start_t2 = time()
        x = self.__calx(x, zwrap)
        #print('cpu_calx_time:' + str(time()-start_t2))

        # Generate the y variables
        y = z - x
        return x, y
    
    
    @jit(nopython=True)
    def __calx(self, x:np.mat, zwrap:np.mat) -> np.mat:
        """ calculate x for model III

        Args:
            x (np.mat): large scale activity
            zwrap (np.mat): vector to assist calculation

        Returns:
            np.mat: large scale activity
        """
        for i in range(self.ss2, self.ss2 + self.model_size):
            x[i - self.ss2, 0] = self.a[0] * zwrap[i + 1 - (- self.smooth_steps), 0] / 2.00
            for j in range(- self.smooth_steps + 1, self.smooth_steps):
                x[i - self.ss2, 0] = x[i - self.ss2, 0] + self.a[j + self.smooth_steps] * zwrap[i + 1 - j, 0]
            x[i - self.ss2, 0] = x[i - self.ss2, 0] + self.a[2 * self.smooth_steps] * zwrap[i + 1 - self.smooth_steps, 0] / 2.00
        return x
    
    
    @jit(nopython=True)
    def __calw(self, wx:np.mat, xwrap:np.mat) -> np.mat:
        """ calculate w for model III

        Args:
            wx (np.mat): store intermediate results
            xwrap (np.mat): vector to assist calculation

        Returns:
            np.mat: wx
        """
        # ! Calculate the W's
        for i in range(self.K4, self.K4 + self.model_size):
            wx[i, 0] = xwrap[i - (-self.H), 0] / 2.00
            for j in range(- self.H + 1, self.H):
                wx[i, 0] = wx[i, 0] + xwrap[i - j, 0]

            wx[i, 0] = wx[i, 0] + xwrap[i - self.H, 0] / 2.00
            wx[i, 0] = wx[i, 0] / self.K
        return wx
    
    
    @jit(nopython=True)
    def __caldz(self, wx:np.mat, xwrap:np.mat, dz:np.mat, ywrap:np.mat) -> np.mat:
        """ calculate time derivative of z

        Args:
            wx (np.mat): store intermediate results
            xwrap (np.mat): vector to assist calculation
            dz (np.mat): time derivative of z
            ywrap (np.mat): vector to assist calculation

        Returns:
            np.mat: dz
        """ 
        for i in range(self.K4, self.K4 + self.model_size):
            xx = wx[i - self.K + (-self.H), 0] * xwrap[i + self.K + (-self.H), 0] / 2.00
            for j in range(- self.H + 1, self.H):
                xx = xx + wx[i - self.K + j, 0] * xwrap[i + self.K + j, 0]
            xx = xx + wx[i - self.K + self.H, 0] * xwrap[i + self.K + self.H, 0] / 2.00
            xx = - wx[i - self.K2, 0] * wx[i - self.K, 0] + xx / self.K

            if self.model_number == 3:
                dz[i - self.K4, 0] = xx + self.sts2 * (- ywrap[i - 2, 0] * ywrap[i - 1, 0] + ywrap[i - 1, 0] * ywrap[i + 1, 0])\
                                + self.coupling * (- ywrap[i - 2, 0] * xwrap[i - 1, 0] + ywrap[i - 1, 0] * xwrap[i + 1, 0]) - xwrap[i, 0]\
                                - self.space_time_scale * ywrap[i, 0] + self.forcing
            else:  # must be model II
                dz[i - self.K4, 0] = xx - xwrap[i, 0] + self.forcing

        return dz

    
    
    
    
    