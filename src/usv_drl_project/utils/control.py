# utils/control.py
import math
import numpy as np
from utils.angle import ssa

dt = 0.1

# Otter USV system matrices
M = np.array([
    [85.2815,  0,      0,      0,      -11,       0],
    [0,      162.5,    0,     11,        0,      11],
    [0,        0,    135,      0,      -11,       0],
    [0,       11,      0,     15.0775,   0,       2.5523],
    [-11,      0,    -11,      0,       31.5184,  0],
    [0,       11,      0,      2.5523,   0,      41.4451]
])

Binv = np.array([
    [45.1264,   114.2439],
    [45.1264,  -114.2439]
])

# Reference model parameters
wn_d = 1.0                      # Natural frequency (rad/s)
zeta_d = 1.0                    # Relative damping factor (-)
r_max = np.deg2rad(10.0)           # Maximum turning rate (rad/s)

# PID heading autopilot parameters (Nomoto model: M(6,6) = T/K)
T = 1                           # Nomoto time constant
K = T / M[5,5]                 # Nomoto gain constant

wn = 1.5                        # Closed-loop natural frequency (rad/s)
zeta = 1.0                      # Closed-loop relative damping factor (-)

Kp = M[5,5] * wn**2                     # Proportional gain
Kd = M[5,5] * (2 * zeta * wn - 1/T)    # Derivative gain
Td = Kd / Kp                           # Derivative time constant
Ti = 10.0 / wn                           # Integral time constant

# Propeller dynamics
T_n = 0.1                       # Propeller time constant (s)

def yaw_pid_controller(psi, psi_d, r, a_d, r_d, z_psi):
    
    psi_error = ssa(psi - psi_d)
    error_d = r - r_d
    error_i = z_psi

    tau_X = 100
    tau_N = (T/K) * a_d + (1/K) * r_d - Kp * (psi_error + Td * error_d + (1/Ti) * error_i)

    z_psi += dt * psi_error

    return tau_X, tau_N, z_psi

def control_allocation(tau_X, tau_N, n):

    u = Binv @ np.array([tau_X, tau_N])
    n_c = np.sign(u) * np.sqrt(np.abs(u))

    n += dt/T_n * (n_c - n)

    return n_c, n