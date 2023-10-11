import numpy as np
from numba import jit, njit, prange
from .step_L04 import gpu


class Lorenz05_gpu:
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
    time_steps = 200 * 360              # 360 days(1 day~ 200 time steps)
    model_number = 3                    # 2 scale
    
    parameter_list = [
        'model_size',
        'forcing',
        'space_time_scale',
        'coupling',
        'smooth_steps', 
        'K',
        'delta_t',
        'time_step_days',
        'time_step_seconds',
        'time_steps',
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
        
        zens_temp = zens
        zens_save = zens
        dzens = self.__comp_dt_L04(zens_temp)  # Compute the first intermediate step
        zens1 = np.multiply(self.delta_t, dzens)
        zens_temp = zens_save + zens1 / 2.0

        dzens = self.__comp_dt_L04(zens_temp)  # Compute the second intermediate step
        zens2 = np.multiply(self.delta_t, dzens)
        zens_temp = zens_save + zens2 / 2.0

        dzens = self.__comp_dt_L04(zens_temp)  # Compute the third intermediate step
        zens3 = np.multiply(self.delta_t, dzens)
        zens_temp = zens_save + zens3

        dzens = self.__comp_dt_L04(zens_temp)  # Compute fourth intermediate step
        zens4 = np.multiply(self.delta_t, dzens)

        dzenst = zens1 / 6.0 + zens2 / 3.0 + zens3 / 3.0 + zens4 / 6.0
        zens_temp = zens_save + dzenst
        
        zens = zens_temp
            
        # update model advance step counter
        self.advance_step_counter += 1
        
        return zens
    
    # private methods
    def __comp_dt_L04(self, zens:np.mat) -> np.mat:
        """ compute the time derivative of the model

        Args:
            zens (np.mat): ensemble state of the model (40 x 960, for example)
            
        Raises:
            ValueError: do not know that model number (model_number != 2 or 3)

        Returns:
            np.mat: time derivative of the model
        """
        if self.model_number == 3:
            xens, yens = self.__z2xy(zens)
        elif self.model_number == 2:
            xens = zens
            yens = 0.0 * zens
        else:
            # print('Do not know that model number')
            raise ValueError(f'Invalid model number: {self.model_number}, should be 2 or 3')
            # return

        #  Deal with  # cyclic boundary# conditions using buffers

        ensemble_size = zens.shape[0]
        # Fill the xwrap and ywrap buffers
        xens_wrap = np.concatenate([xens[:, self.model_size - self.K4: self.model_size], xens, xens[:, 0: self.K4]], axis=1)
        yens_wrap = np.concatenate([yens[:, self.model_size - self.K4: self.model_size], yens, yens[:, 0: self.K4]], axis=1)

        wxens = np.mat(np.zeros((ensemble_size, self.model_size + self.K4 * 2)))
        # ! Calculate the W's
        wxens = self.__calw(wxens, xens_wrap)

        # Fill the W buffers
        wxens[:, 0: self.K4] = wxens[:, self.model_size: self.model_size + self.K4]
        wxens[:, self.model_size + self.K4: self.model_size + 2 * self.K4] = wxens[:, self.K4: self.K4 * 2]

        dzens = np.mat(np.zeros((ensemble_size, self.model_size)))
        # ! Generate dzens / dt
        dzens = self.__caldz(wxens, xens_wrap, dzens, yens_wrap)

        return dzens
    
    
    def __z2xy(self, zens:np.mat) -> (np.mat, np.mat):
        """ convert zens to xens and yens for model III

        Args:
            zens (np.mat): ensemble state of the model

        Returns:
            (np.mat, np.mat): x and y
            xens: large scale activity (ensemble)
            yens: small scale activity (ensemble)
            zens = xens + yens
        """
        # ss2 is smoothing scale I * 2
        # Fill zwrap
        zens_wrap = np.concatenate([zens[:, (self.model_size - self.ss2 - 1): self.model_size], zens, zens[:, 0: self.ss2]], axis=1)
        xens = np.mat(np.zeros(zens.shape))
        yens = np.mat(np.zeros(zens.shape))

        # CPU version -- Generate the x variables
        #start_t2 = time()
        xens = self.__calx(xens, zens_wrap)
        #print('cpu_calx_time:' + str(time()-start_t2))
        # assert np.nan not in xens
        # print('xens:', xens)

        # Generate the y variables
        yens = zens - xens
        return xens, yens
    
    
    def __calx(self, xens:np.mat, zens_wrap:np.mat) -> np.mat:
        # return calx(xens, zens_wrap, self.a, self.model_size, self.ss2, self.smooth_steps)
        return gpu.calx(xens, zens_wrap, self.a, self.model_size, self.ss2, self.smooth_steps)
    
    
    def __calw(self, wxens:np.mat, xens_wrap:np.mat) -> np.mat:
        # return calw(wxens, xens_wrap, self.K, self.K4, self.H, self.model_size)
        return gpu.calw(wxens, xens_wrap, self.K, self.K4, self.H, self.model_size)
    
    
    def __caldz(self, wxens:np.mat, xens_wrap:np.mat, dzens:np.mat, yens_wrap:np.mat) -> np.mat:
        # return caldz(wxens, xens_wrap, dzens, yens_wrap, self.space_time_scale, self.sts2, self.coupling, self.forcing, self.K, self.K2, self.K4, self.H, self.model_size, self.model_number)
        return gpu.caldz(wxens, xens_wrap, dzens, yens_wrap, self.space_time_scale, self.sts2, self.coupling, self.forcing, self.K, self.K2, self.K4, self.H, self.model_size, self.model_number)
    

@jit(nopython=True)
def calx(xens:np.mat, zens_wrap:np.mat, a:np.mat, model_size:int, ss2:float, smooth_steps:float) -> np.mat:
    """ calculate x for model III

    Args:
        x (np.mat): large scale activity
        zwrap (np.mat): vector to assist calculation

    Returns:
        np.mat: large scale activity
    """
    for i in range(ss2, ss2 + model_size):
        xens[:, i - ss2] = a[0] * zens_wrap[:, i + 1 - (- smooth_steps)] / 2.00
        for j in range(- smooth_steps + 1, smooth_steps):
            xens[:, i - ss2] = xens[:, i - ss2] + a[j + smooth_steps] * zens_wrap[:, i + 1 - j]
        xens[:, i - ss2] = xens[:, i - ss2] + a[2 * smooth_steps] * zens_wrap[:, i + 1 - smooth_steps] / 2.00
    return xens


@jit(nopython=True)
def calw(wxens:np.mat, xens_wrap:np.mat, K:int, K4:int, H:int, model_size:int) -> np.mat:
    """ calculate w for model III

    Args:
        wx (np.mat): store intermediate results
        xwrap (np.mat): vector to assist calculation

    Returns:
        np.mat: wx
    """
    # ! Calculate the W's
    for i in range(K4, K4 + model_size):
        wxens[:, i] = xens_wrap[:, i - (-H)] / 2.00
        for j in range(- H + 1, H):
            wxens[:, i] = wxens[:, i] + xens_wrap[:, i - j]

        wxens[:, i] = wxens[:, i] + xens_wrap[:, i - H] / 2.00
        wxens[:, i] = wxens[:, i] / K
    return wxens


@jit(nopython=True)
def caldz(wxens:np.mat, xens_wrap:np.mat, dzens:np.mat, yens_wrap:np.mat, space_time_scale:float, sts2:float, coupling:float, 
          forcing:float, K:int, K2:int, K4:int, H:int, model_size:int, model_number:int) -> np.mat:
    """ calculate time derivative of z

    Args:
        wx (np.mat): store intermediate results
        xwrap (np.mat): vector to assist calculation
        dz (np.mat): time derivative of z
        ywrap (np.mat): vector to assist calculation

    Returns:
        np.mat: dz
    """ 
    for i in range(K4, K4 + model_size):
        xx = wxens[:, i - K + (-H)] * xens_wrap[:, i + K + (-H)] / 2.00
        for j in range(- H + 1, H):
            xx = xx + wxens[:, i - K + j] * xens_wrap[:, i + K + j]
        xx = xx + wxens[:, i - K + H] * xens_wrap[:, i + K + H] / 2.00
        xx = - wxens[:, i - K2] * wxens[:, i - K] + xx / K

        if model_number == 3:
            dzens[:, i - K4] = xx + sts2 * (- yens_wrap[:, i - 2] * yens_wrap[:, i - 1] + yens_wrap[:, i - 1] * yens_wrap[:, i + 1])\
                            + coupling * (- yens_wrap[:, i - 2] * xens_wrap[:, i - 1] + yens_wrap[:, i - 1] * xens_wrap[:, i + 1]) - xens_wrap[:, i]\
                            - space_time_scale * yens_wrap[:, i] + forcing
        else:  # must be model II
            dzens[:, i - K4] = xx - xens_wrap[:, i] + forcing

    return dzens
