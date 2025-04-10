import numpy as np
from scipy.interpolate import interp1d

def Hoerner(B, T):

    CD_DATA = np.array([    
    [0.0108623, 1.96608],
    [0.176606, 1.96573],
    [0.353025, 1.89756],
    [0.451863, 1.78718],
    [0.472838, 1.58374],
    [0.492877, 1.27862],
    [0.493252, 1.21082],
    [0.558473, 1.08356],
    [0.646401, 0.998631],
    [0.833589, 0.87959],
    [0.988002, 0.828415],
    [1.30807, 0.759941],
    [1.63918, 0.691442],
    [1.85998, 0.657076],
    [2.31288, 0.630693],
    [2.59998, 0.596186],
    [3.00877, 0.586846],
    [3.45075, 0.585909],
    [3.7379, 0.559877],
    [4.00309, 0.559315]])

    if B/(2*T) <= 4.00309:
        interp_func = interp1d(CD_DATA[:, 0], CD_DATA[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
        CY_2D = interp_func(B/(2*T))
    else:
        CY_2D = 0.559315
    
    return CY_2D