import numpy as np

def ssa(angle, unit=None):
    if unit == 'deg':
        angle = (angle + 180) % 360 - 180
    else:
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle