import numpy as np
from numpy.linalg import inv, solve
from utils.kinematics import Smtrx, Hmtrx, Rzyx, eulerang
from utils.modeling import added_mass_surge, cross_flow_drag, m2c

# def otter(x,n,mp,rp,V_c,beta_c):
def otter(*args, **kwargs):

    if len(args) == 0 and len(kwargs) == 0:
        x = np.zeros(12)
        n = np.zeros(2)
        mp = 25.0
        rp = np.zeros(3)
        V_c = 0.0
        beta_c = 0.0
    elif len(args) == 6:
        x, n, mp, rp, V_c, beta_c = args
    else:
        x = kwargs.get('x')
        n = kwargs.get('n')
        mp = kwargs.get('mp')
        rp = kwargs.get('rp')
        V_c = kwargs.get('V_c')
        beta_c = kwargs.get('beta_c')

    # Check of input and state dimensions
    if len(x) != 12:
        raise ValueError("x vector must have dimension 12!") 
    if len(n) != 2:
        raise ValueError("n vector must have dimension 2!") 

    # Main data
    g   = 9.81         # acceleration of gravity (m/s**2)
    rho = 1025         # density of water
    L = 2.0            # len (m)
    B = 1.08           # beam (m)
    m = 55.0           # mass (kg)
    rg = np.array([0.2, 0, -0.2]) # CG for hull only (m)
    R44 = 0.4 * B      # radii of gyrations (m)
    R55 = 0.25 * L
    R66 = 0.25 * L
    T_sway = 1         # time constant in sway (s)
    T_yaw = 1          # time constant in yaw (s)
    Umax = 6 * 0.5144  # 6 knots maximum forward speed (m/s)

    # Data for one pontoon
    B_pont  = 0.25     # beam of one pontoon (m)
    y_pont  = 0.395    # distance from centerline to waterline area center (m)
    Cw_pont = 0.75     # waterline area coefficient (-)
    Cb_pont = 0.4      # block coefficient, computed from m = 55 kg

    # State and current variables
    nu = x[0:6]  
    nu2 = x[3:6]   # velocity vectors
    eta = x[6:12]                              # positions
    U = np.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)      # speed
    u_c = V_c * np.cos(beta_c - eta[5])           # current surge velocity
    v_c = V_c * np.sin(beta_c - eta[5])           # current sway velocity
    nu_c = np.array([u_c, v_c, 0, 0, 0, 0])                 # current velocity vector
    nu_r = nu - nu_c                           # relative velocity vector
    nu_c_dot1 = -Smtrx(nu2) @ nu_c[0:3]
    nu_c_dot2 = np.zeros(3)
    nu_c_dot = np.concatenate([nu_c_dot1, nu_c_dot2])

    # Inertia dyadic, volume displacement and draft
    nabla = (m + mp)/rho                         # volume
    T = nabla / (2 * Cb_pont * B_pont * L)     # draft
    Ig_CG = m * np.diag([R44**2, R55**2, R66**2])    # only hull in the CG
    rg = (m*rg + mp*rp)/(m + mp)           # CG location corrected for payload
    Ig = Ig_CG - m * (Smtrx(rg) @ Smtrx(rg)) - mp * (Smtrx(rp) @ Smtrx(rp))

    # Experimental propeller data including lever arms
    l1 = -y_pont                           # lever arm, left propeller (m)
    l2 = y_pont                            # lever arm, right propeller (m)
    k_pos = 0.02216/2                      # Positive Bollard, one propeller 
    k_neg = 0.01289/2                      # Negative Bollard, one propeller 
    n_max =  np.sqrt((0.5*24.4 * g)/k_pos)    # maximum propeller rev. (rad/s)
    n_min = -np.sqrt((0.5*13.6 * g)/k_neg)    # minimum propeller rev. (rad/s)

    # MRB and CRB (Fossen 2021)
    I3 = np.eye(3)
    O3 = np.zeros((3,3))

    MRB_CG = np.block([
        [(m + mp) * I3,    O3],
        [      O3     ,    Ig]
    ])

    CRB_CG = np.block([
        [(m + mp) * Smtrx(nu2),               O3],
        [        O3           , -Smtrx(Ig @ nu2)]
    ])


    H = Hmtrx(rg)              # Transform MRB and CRB from the CG to the CO 
    MRB = H.T @ MRB_CG @ H
    CRB = H.T @ CRB_CG @ H

    # Hydrodynamic added mass (best practice)
    Xudot = -added_mass_surge(m, L, rho)[0]
    Yvdot = -1.5 * m
    Zwdot = -1.0 * m
    Kpdot = -0.2 * Ig[0,0]
    Mqdot = -0.8 * Ig[1,1]
    Nrdot = -1.7 * Ig[2,2]

    MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])   
    CA  = m2c(MA, nu_r)

    # Uncomment to cancel the Munk moment in yaw, if stability problems
    # CA(6,1) = 0 
    # CA(6,2) = 0 
    # CA(1,6) = 0
    # CA(2,6) = 0

    # System mass and Coriolis-centripetal matrices
    M = MRB + MA
    C = CRB + CA

    # Hydrostatic quantities (Fossen 2021)
    Aw_pont = Cw_pont * L * B_pont    # waterline area, one pontoon 
    I_T = 2 * (1/12)*L*B_pont**3 * (6*Cw_pont**3/((1+Cw_pont)*(1+2*Cw_pont))) + 2 * Aw_pont * y_pont**2
    I_L = 0.8 * 2 * (1/12) * B_pont * L**3
    KB = (1/3)*(5*T/2 - 0.5*nabla/(L*B_pont) )
    BM_T = I_T/nabla       # BM values
    BM_L = I_L/nabla
    KM_T = KB + BM_T       # KM values
    KM_L = KB + BM_L
    KG = T - rg[2]
    GM_T = KM_T - KG       # GM values
    GM_L = KM_L - KG

    G33 = rho * g * (2 * Aw_pont)      # spring stiffness
    G44 = rho * g * nabla * GM_T
    G55 = rho * g * nabla * GM_L

    G_CF = np.diag([0, 0, G33, G44, G55, 0])   # spring stiffness matrix in the CF
    LCF = -0.2
    H = Hmtrx(np.array([LCF, 0, 0]))
    G = H.T @ G_CF @ H

    # Natural frequencies
    w3 = np.sqrt(G33/M[2,2])
    w4 = np.sqrt(G44/M[3,3])
    w5 = np.sqrt(G55/M[4,4])

    # Linear damping terms (hydrodynamic derivatives)
    Xu = -24.4 * g / Umax        # specified using the maximum speed  
    Yv = -M[1,1] / T_sway        # specified using the time constant in sway
    Zw = -2 * 0.3 * w3 * M[2,2]  # specified using relative damping factors
    Kp = -2 * 0.2 * w4 * M[3,3]
    Mq = -2 * 0.4 * w5 * M[4,4]
    Nr = -M[5,5] / T_yaw         # specified using the time constant in T_yaw

    # 2-DOF constant input matrix B_prop for the propellers i sway and yaw,
    # accessable by: [~,~,M, B_prop] = otter()
    if len(args) == 0 and len(kwargs) == 0:
        B_prop = k_pos * np.vstack(([1, 1], [y_pont, -y_pont]))
    else: B_prop = []

    # Control forces and moments, with saturated propeller speed
    n_saturated = np.clip(n, n_min, n_max)
    Thrust = np.zeros(2)

    for i in range(2):
        if n_saturated[i] > 0:
            Thrust[i] = k_pos * n_saturated[i] * np.abs(n_saturated[i])
        else:
            Thrust[i] = k_neg * n_saturated[i] * np.abs(n_saturated[i])

    # Control forces and moments
    tau = np.array([Thrust[0] + Thrust[1], 0, 0, 0, 0, -l1 * Thrust[0] - l2 * Thrust[1]])

    # Linear damping using relative velocities + nonlinear yaw damping
    Xh = Xu * nu_r[0]
    Yh = Yv * nu_r[1] 
    Zh = Zw * nu_r[2]
    Kh = Kp * nu_r[3]
    Mh = Mq * nu_r[4]
    Nh = Nr * (1 + 10 * np.abs(nu_r[5])) * nu_r[5]

    tau_damp = np.array([Xh, Yh, Zh, Kh, Mh, Nh])

    # Strip theory: cross-flow drag integrals
    tau_crossflow = cross_flow_drag(L,B_pont,T,nu_r)

    # Payload expressed in NED
    f_payload = Rzyx(eta[3],eta[4],eta[5]).T @ np.array([0, 0, mp*g])  # payload force 
    m_payload = Smtrx(rp) @ f_payload                        # payload moment 
    g_0 = np.concatenate((f_payload, m_payload))

    # Trim condition: G * eta_0 = g_0
    eta_0 = np.zeros(6)
    eta_0[2:5] = inv(G[2:5, 2:5]) @ g_0[2:5]
    eta = eta - eta_0 # shifted equilibrium

    # Kinematic transformation matrix
    J = eulerang(eta[3],eta[4],eta[5])[0]

    # Time derivative of the state vector, numerical integration see ExOtter.m   
    nu_rhs = tau + tau_damp + tau_crossflow - C @ nu_r - G @ eta
    nu_dot = nu_c_dot + solve(M, nu_rhs) 
    eta_dot = J @ nu

    xdot = np.concatenate((nu_dot, eta_dot))

    return xdot, U, M, B_prop