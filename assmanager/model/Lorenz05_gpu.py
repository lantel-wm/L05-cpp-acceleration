# import numpy as np
# import cupy as cp


# class Lorenz05_gpu:
#     """ Lorenz05 model

#     Raises:
#         ValueError: Invalid parameter
#             in __init__(), if the parameter is not in parameter_list
#         ValueError: Invalid model number
#             in __comp_dt_L04(), if the model number is not 2 or 3
#     """
#     # default parameters
#     # model parameters
#     model_size = 960                    # N
#     forcing = 15.00                     # F
#     space_time_scale = 10.00            # b
#     coupling = 3.00                     # c
#     smooth_steps = 12                   # I
#     K = 32
#     delta_t = 0.001
#     time_step_days = 0
#     time_step_seconds = 432
#     time_steps = 200 * 360              # 360 days(1 day~ 200 time steps)
#     model_number = 3                    # 2 scale
    
#     parameter_list = [
#         'model_size',
#         'forcing',
#         'space_time_scale',
#         'coupling',
#         'smooth_steps', 
#         'K',
#         'delta_t',
#         'time_step_days',
#         'time_step_seconds',
#         'time_steps',
#         'model_number',
#     ]
    
    
#     def __init__(self, params:dict) -> None:
#         """ model parameters initialization

#         Args:
#             parameters (dict): parameters of the model

#         Raises:
#             ValueError: invalid parameter
#         """
#         for key in params:
#             if key not in self.parameter_list:
#                 raise ValueError(f'Invalid parameter: {key}')
#             setattr(self, key, params[key])
            
#         # for speed
#         self.H = int(self.K / 2)
#         self.K2 = 2 * self.K
#         self.K4 = 4 * self.K
#         self.ss2 = 2 * self.smooth_steps
#         self.sts2 = self.space_time_scale ** 2

#         # smoothing filter
#         self.alpha = (3.0 * (self.smooth_steps ** 2) + 3.0) / (2.0 * (self.smooth_steps ** 3) + 4.0 * self.smooth_steps)
#         self.beta = (2.0 * (self.smooth_steps ** 2) + 1.0) / (1.0 * (self.smooth_steps ** 4) + 2.0 * (self.smooth_steps ** 2))

#         ri = - self.smooth_steps - 1.0
#         j = 0
#         self.a = np.zeros(2*self.smooth_steps+1)
#         for i in range(-self.smooth_steps, self.smooth_steps+1):
#             ri = ri + 1.0
#             self.a[j] = self.alpha - self.beta * abs(ri)
#             j = j + 1

#         self.a[0] = self.a[0] / 2.00
#         self.a[2 * self.smooth_steps] = self.a[2 * self.smooth_steps] / 2.00
        
#         # model advance step counter (istep)
#         self.advance_step_counter = 0
    
        
#     # public method    
#     def step_L04(self, zens:np.mat) -> np.mat: 
#         """ integrate the model for one time step

#         Args:
#             z (cp.ndarray): ensemble state of the model
            
#         Returns:
#             cp.ndarray: state of the model after integration
#         """
        
#         zens_cupy = cp.asarray(zens)
        
#         for iens in range(zens_cupy.shape[0]):
#             z = zens_cupy[iens, :]
            
#             z_save = z
#             dz = self.__comp_dt_L04(z)  # Compute the first intermediate step
#             z1 = self.delta_t * dz
#             z = z_save + z1 / 2.0

#             dz = self.__comp_dt_L04(z)  # Compute the second intermediate step
#             z2 = self.delta_t * dz
#             z = z_save + z2 / 2.0

#             dz = self.__comp_dt_L04(z)  # Compute the third intermediate step
#             z3 = self.delta_t * dz
#             z = z_save + z3

#             dz = self.__comp_dt_L04(z)  # Compute fourth intermediate step
#             z4 = self.delta_t * dz

#             dzt = z1 / 6.0 + z2 / 3.0 + z3 / 3.0 + z4 / 6.0
#             z = z_save + dzt
            
#             zens_cupy[iens, :] = z
            
#         zens = np.mat(cp.asnumpy(zens_cupy))
            
#         # update model advance step counter
#         self.advance_step_counter += 1
        
#         return zens
    
#     # private methods
#     def __comp_dt_L04(self, z:cp.ndarray) -> cp.ndarray:
#         """ compute the time derivative of the model

#         Args:
#             z (cp.ndarray): state of the model
            
#         Raises:
#             ValueError: do not know that model number (model_number != 2 or 3)

#         Returns:
#             cp.ndarray: time derivative of the model
#         """
#         if self.model_number == 3:
#             x, y = self.__z2xy(z)
#         elif self.model_number == 2:
#             x = z
#             y = 0.0 * z
#         else:
#             # print('Do not know that model number')
#             raise ValueError(f'Invalid model number: {self.model_number}, should be 2 or 3')
#             # return

#         #  Deal with  # cyclic boundary# conditions using buffers

#         # Fill the xwrap and ywrap buffers
#         xwrap = cp.concatenate([x[self.model_size - self.K4: self.model_size], x, x[0: self.K4]])
#         ywrap = cp.concatenate([y[self.model_size - self.K4: self.model_size], y, y[0: self.K4]])

#         wx = cp.zeros(self.model_size + self.K4 * 2)
#         # ! Calculate the W's
#         wx = self.__calw(wx, xwrap)

#         # Fill the W buffers
#         wx[0: self.K4] = wx[self.model_size: self.model_size + self.K4]
#         wx[self.model_size + self.K4: self.model_size + 2 * self.K4] = wx[self.K4: self.K4 * 2]

#         dz = cp.zeros(self.model_size)
#         # ! Generate dz / dt
#         dz = self.__caldz(wx, xwrap, dz, ywrap)

#         return dz
    
    
#     def __z2xy(self, z:cp.ndarray) -> (cp.ndarray, cp.ndarray):
#         """ convert z to x and y for model III

#         Args:
#             z (cp.ndarray): state of the model

#         Returns:
#             (cp.ndarray, cp.ndarray): x and y
#             x: large scale activity
#             y: small scale activity
#             z = x + y
#         """
#         # ss2 is smoothing scale I * 2
#         # Fill zwrap
#         # print('z.shape: ', z.shape)
#         zwrap = cp.concatenate([z[(self.model_size - self.ss2 - 1): self.model_size], z, z[0: self.ss2]])
#         x = cp.zeros(self.model_size)
#         y = cp.zeros(self.model_size)

#         # CPU version -- Generate the x variables
#         #start_t2 = time()
#         x = self.__calx(x, zwrap)
#         #print('cpu_calx_time:' + str(time()-start_t2))

#         # Generate the y variables
#         y = z - x
#         return x, y
    
    
#     def __calx(self, x:cp.ndarray, zwrap:cp.ndarray) -> cp.ndarray:
#         return calx(x, zwrap, self.a, self.model_size, self.ss2, self.smooth_steps)
    
    
#     def __calw(self, wx:cp.ndarray, xwrap:cp.ndarray) -> cp.ndarray:
#         return calw(wx, xwrap, self.K, self.K4, self.H, self.model_size)
    
    
#     def __caldz(self, wx:cp.ndarray, xwrap:cp.ndarray, dz:cp.ndarray, ywrap:cp.ndarray) -> cp.ndarray:
#         return caldz(wx, xwrap, dz, ywrap, self.space_time_scale, self.sts2, self.coupling, self.forcing, self.K, self.K2, self.K4, self.H, self.model_size, self.model_number)

    
# def calx(x:cp.ndarray, zwrap:cp.ndarray, a:cp.ndarray, model_size:int, ss2:float, smooth_steps:float) -> cp.ndarray:
#     """ calculate x for model III

#     Args:
#         x (cp.ndarray): large scale activity
#         zwrap (cp.ndarray): vector to assist calculation

#     Returns:
#         cp.ndarray: large scale activity
#     """
#     # print(x.shape)
#     # print(zwrap.shape)
#     # print(a.shape)
#     for i in range(ss2, ss2 + model_size):
#         x[i - ss2] = a[0] * zwrap[i + 1 - (- smooth_steps)] / 2.00
#         for j in range(- smooth_steps + 1, smooth_steps):
#             x[i - ss2] = x[i - ss2] + a[j + smooth_steps] * zwrap[i + 1 - j]
#         x[i - ss2] = x[i - ss2] + a[2 * smooth_steps] * zwrap[i + 1 - smooth_steps] / 2.00
#     return x


# def calw(wx:cp.ndarray, xwrap:cp.ndarray, K:int, K4:int, H:int, model_size:int) -> cp.ndarray:
#     """ calculate w for model III

#     Args:
#         wx (cp.ndarray): store intermediate results
#         xwrap (cp.ndarray): vector to assist calculation

#     Returns:
#         cp.ndarray: wx
#     """
#     # ! Calculate the W's
#     for i in range(K4, K4 + model_size):
#         wx[i] = xwrap[i - (-H)] / 2.00
#         for j in range(- H + 1, H):
#             wx[i] = wx[i] + xwrap[i - j]

#         wx[i] = wx[i] + xwrap[i - H] / 2.00
#         wx[i] = wx[i] / K
#     return wx


# def caldz(wx:cp.ndarray, xwrap:cp.ndarray, dz:cp.ndarray, ywrap:cp.ndarray, space_time_scale:float, sts2:float, coupling:float, 
#           forcing:float, K:int, K2:int, K4:int, H:int, model_size:int, model_number:int) -> cp.ndarray:
#     """ calculate time derivative of z

#     Args:
#         wx (cp.ndarray): store intermediate results
#         xwrap (cp.ndarray): vector to assist calculation
#         dz (cp.ndarray): time derivative of z
#         ywrap (cp.ndarray): vector to assist calculation

#     Returns:
#         cp.ndarray: dz
#     """ 
#     for i in range(K4, K4 + model_size):
#         xx = wx[i - K + (-H)] * xwrap[i + K + (-H)] / 2.00
#         for j in range(- H + 1, H):
#             xx = xx + wx[i - K + j] * xwrap[i + K + j]
#         xx = xx + wx[i - K + H] * xwrap[i + K + H] / 2.00
#         xx = - wx[i - K2] * wx[i - K] + xx / K

#         if model_number == 3:
#             dz[i - K4] = xx + sts2 * (- ywrap[i - 2] * ywrap[i - 1] + ywrap[i - 1] * ywrap[i + 1])\
#                             + coupling * (- ywrap[i - 2] * xwrap[i - 1] + ywrap[i - 1] * xwrap[i + 1]) - xwrap[i]\
#                             - space_time_scale * ywrap[i] + forcing
#         else:  # must be model II
#             dz[i - K4] = xx - xwrap[i] + forcing

#     return dz
