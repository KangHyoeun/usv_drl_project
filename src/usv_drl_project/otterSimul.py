import numpy as np
from numpy import unwrap
import matplotlib.pyplot as plt
from models.otter import otter
from utils.gnc import ref_model

"""
This script simulates the Otter Uncrewed Surface Vehicle (USV) 

The simulation covers:
1. PID heading autopilot without path following.

Dependencies:
otter                 - Dynamics of the Otter USV
ref_model              - Reference model for autopilot systems
"""

def ssa(angle, unit=None):
    if unit == 'deg':
        angle = (angle + 180) % 360 - 180
    else:
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

def rk4(func, h, x, *args):
    """Runge-Kutta 4th order method"""
    x = np.asarray(x)
    k1 = func(x, *args)
    k1 = np.asarray(k1)
    k2 = func(x + h / 2 * k1, *args)
    k2 = np.asarray(k2)
    k3 = func(x + h / 2 * k2, *args)
    k3 = np.asarray(k3)
    k4 = func(x + h * k3, *args)
    k4 = np.asarray(k4)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    
if __name__ == "__main__":
    ## USER INPUTS
    h  = 0.05                 # Sampling time [s]
    T_final = 1000	                 # Final simulation time [s]

    # Load condition
    mp = 25                         # Payload mass (kg), maximum value 45 kg
    rp = np.array([0.05, 0, -0.35])            # Location of payload (m)

    # Ocean current
    V_c = 0.3                       # Ocean current speed (m/s)
    beta_c = np.deg2rad(30)            # Ocean current direction (rad)

    # Additional parameter for straight-line path following
    R_switch = 5                    # Radius of switching circle
    K_f = 0.3                       # LOS observer gain

    # Initial heading, vehicle points towards next waypoint
    psi0 = np.radians(90)
    
    print(f"psi0 = {np.rad2deg(psi0):.2f}")

    # Otter USV input matrix
    M = np.array([
        [85.2815,  0,      0,      0,      -11,       0],
        [0,      162.5,    0,     11,        0,      11],
        [0,        0,    135,      0,      -11,       0],
        [0,       11,      0,     15.0775,   0,       2.5523],
        [-11,      0,    -11,      0,       31.5184,  0],
        [0,       11,      0,      2.5523,   0,      41.4451]
    ])
    B_prop = np.array([
        [0.0111,     0.0111],
        [0.0044,  -0.0044]
    ])

    Binv = np.array([
        [45.1264,   114.2439],
        [45.1264,  -114.2439]
    ])

    # PID heading autopilot parameters (Nomoto model: M(6,6) = T/K)
    T = 1                           # Nomoto time constant
    K = T / M[5,5]                 # Nomoto gain constant

    wn = 1.5                        # Closed-loop natural frequency (rad/s)
    zeta = 1.0                      # Closed-loop relative damping factor (-)

    Kp = M[5,5] * wn**2                     # Proportional gain
    Kd = M[5,5] * (2 * zeta * wn - 1/T)    # Derivative gain
    Td = Kd / Kp                           # Derivative time constant
    Ti = 10.0 / wn                           # Integral time constant

    # Reference model parameters
    wn_d = 1.0                      # Natural frequency (rad/s)
    zeta_d = 1.0                    # Relative damping factor (-)
    r_max = np.deg2rad(10.0)           # Maximum turning rate (rad/s)

    # Propeller dynamics
    T_n = 0.1                       # Propeller time constant (s)
    n = np.array([0.0, 0.0])                      # Initial propeller speed, [n_left n_right]'

    # Initial states
    x = np.zeros(12)                 # x = [u v w p q r xn yn zn phi theta psi]'
    x[11] = psi0                    # Heading angle
    z_psi = 0.0                       # Integral state for heading control
    psi_d = psi0                    # Desired heading angle
    r_d = 0.0                         # Desired rate of turn
    a_d = 0.0                         # Desired acceleration

    # Time vector initialization
    t = np.arange(0.0, T_final + h, h)  # Time vector from 0 to T_final
    nTimeSteps = len(t)         # Number of time steps

    ## MAIN LOOP
    simdata = np.zeros((nTimeSteps, 14))    # Preallocate table for simulation data

    for i in range(nTimeSteps):

        # Measurements with noise
        r = x[5] + 0.001 * np.random.randn()       # Yaw rate 
        xn = x[6] + 0.01 * np.random.randn()       # North position
        yn = x[7] + 0.01 * np.random.randn()       # East position
        psi = x[11] + 0.001 * np.random.randn()    # Yaw angle

        if t[i] > 500:
            psi_ref = np.deg2rad(-90)
        elif t[i] > 100:
            psi_ref = np.deg2rad(0)
        else:
            psi_ref = psi0

        # Reference model propagation
        psi_d, r_d, a_d = ref_model(psi_d, r_d, a_d, psi_ref, r_max, zeta_d, wn_d, h, 1)
        
        # PID heading (yaw moment) autopilot and forward thrust
        tau_X = 100                              # Constant forward thrust
        tau_N = (T/K) * a_d + (1/K) * r_d - Kp * (ssa(psi - psi_d) + Td * (r - r_d) + (1/Ti) * z_psi) # Derivative and integral terms

        # Control allocation
        u = Binv @ np.array([tau_X, tau_N])      # Compute control inputs for propellers
        n_c = np.sign(u) * np.sqrt(np.abs(u))  # Convert to required propeller speeds

        # Debug log every 5 steps
        if i % 50 == 0:
            print(f"[t={t[i]:.1f}s] tau_N = {tau_N:.2f}, psi = {np.rad2deg(psi):.2f}, psi_d = {np.rad2deg(psi_d):.2f}")
            print(f"         u = {u}, n_c = {n_c}, n = {n}")
            print(f"         ref_model: psi_d = {np.rad2deg(psi_d):.2f}, r_d = {np.rad2deg(r_d):.2f}, a_d = {np.rad2deg(a_d):.2f}")

        # Store simulation data
        simdata[i, :] = np.concatenate((x.T, [r_d, psi_d]))

        # RK4 method x(k+1)
        x = rk4(lambda x_, *args: otter(x_, *args)[0], h, x, n, mp, rp, V_c, beta_c)

        # Euler's method
        n = n + h/T_n * (n_c - n)              # Update propeller speeds
        z_psi = z_psi + h * ssa(psi - psi_d)   # Update integral state


    ## PLOTS

    # Simulation data structure
    nu   = simdata[:,0:6] 
    eta  = simdata[:,6:12] 
    r_d = simdata[:,12]    
    psi_d = simdata[:,13]  

    # positions
    plt.figure(1, figsize=(10, 10))
    plt.plot(eta[:,1],eta[:,0],'b', label='Vehicle position')  # vehicle position

    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('North-East Positions (m)')

    # plt.plot velocities
    plt.figure(2, figsize=(4, 10))
    plt.subplot(611)
    plt.plot(t,nu[:,0])
    plt.xlabel('Time (s)')
    plt.ylabel('Surge velocity (m/s)')
    plt.subplot(612)
    plt.plot(t,nu[:,1])
    plt.xlabel('Time (s)')
    plt.ylabel('Sway velocity (m/s)')
    plt.subplot(613)
    plt.plot(t,nu[:,2])
    plt.xlabel('Time (s)')
    plt.ylabel('Heave velocity (m/s)')
    plt.subplot(614)
    plt.plot(t,np.rad2deg(nu[:,3]))
    plt.xlabel('Time (s)')
    plt.ylabel('Roll rate (deg/s)')
    plt.subplot(615)
    plt.plot(t,np.rad2deg(nu[:,4]))
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch rate (deg/s)')
    plt.subplot(616)
    plt.plot(t,np.rad2deg(nu[:,5]),label='r')
    plt.plot(t,np.rad2deg(r_d),label='r_d')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw rate (deg/s)')
    plt.legend()

    # plt.plot speed, heave position and Euler angles
    plt.figure(3, figsize=(4, 10))
    plt.subplot(511)
    plt.plot(t, np.sqrt(nu[:,0]**2 + nu[:,1]**2))
    plt.ylabel('Speed (m/s)')
    plt.subplot(512)
    plt.plot(t,eta[:,2])
    plt.ylabel('Heave position (m)')
    plt.subplot(513)
    plt.plot(t,np.rad2deg(eta[:,3]))
    plt.ylabel('Roll angle (deg)')
    plt.subplot(514)
    plt.plot(t,np.rad2deg(eta[:,4]))
    plt.ylabel('Pitch angle (deg)')
    plt.subplot(515)
    plt.plot(t,np.rad2deg(unwrap(eta[:,5])),label='$\psi$')
    plt.plot(t,np.rad2deg(unwrap(psi_d)),label='$\psi_d$')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw angle (deg)')
    plt.legend()

    plt.show()

    # Display the vehicle data and an image of the vehicle
    vehicleName = "Maritime Robotics Otter USV"
    imageFile = "otter.jpg"
    figNo = 4
    vehicleData = [('len', '2.0 m'), ('Beam', '1.08 m'), ('Draft (no payload)', '13.4 cm'), ('Draft (25 kg payload)', '19.5 cm'), ('Mass (no payload)', '55.0 kg'), ('Max speed', '3.0 m/s'), ('Max pos. propeller speed', '993 RPM'), ('Max neg. propeller speed', '-972 RPM')]



