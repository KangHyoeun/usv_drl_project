import numpy as np
from utils.hydro import Hoerner
from utils.kinematics import Hmtrx, Smtrx

def added_mass_surge(m, L, rho=1025):

    nabla = m / rho
    A11 = 2.7 * rho * nabla**(5/3) / L**2
    ratio = A11 / m

    return A11, ratio

def cross_flow_drag(L, B, T, nu_r):

    rho = 1025

    dx = L/20
    Cd_2D = Hoerner(B, T)

    Yh, Zh, Mh, Nh = 0, 0, 0, 0

    x_range = np.arange(-L/2, L/2 + dx, dx)

    for xL in x_range:
        v_r = nu_r[1]
        w_r = nu_r[2]
        q = nu_r[4]
        r = nu_r[5]
        U_h = np.abs(v_r + xL * r) * (v_r + xL * r)
        U_v = np.abs(w_r + xL * q) * (w_r + xL * q)
        Yh -= 0.5 * rho * T * Cd_2D * U_h * dx
        Zh -= 0.5 * rho * T * Cd_2D * U_v * dx
        Mh -= 0.5 * rho * T * Cd_2D * xL * U_v * dx
        Nh -= 0.5 * rho * T * Cd_2D * xL * U_h * dx

    tau_crossflow = np.array([0, Yh, Zh, 0, Mh, Nh])

    return tau_crossflow

def Dmtrx(T_126, zeta_45, MRB, MA, hydrostatics):

    M = MRB + MA

    T1 = T_126[0]
    T2 = T_126[1]
    T6 = T_126[2]
    zeta4 = zeta_45[0]
    zeta5 = zeta_45[1]

    if isinstance(hydrostatics, np.ndarray) and hydrostatics.ndim == 1: # submerged vehicle: hydrostatics = [W, r_bg, r_bb]
        W = hydrostatics[0]
        r_bg = hydrostatics[1:4]
        r_bb = hydrostatics[4:7]

        T3 = T2
        w4 = np.sqrt(W * (r_bg[2] - r_bb[2]) / M[3,3])
        w5 = np.sqrt(W * (r_bg[2] - r_bb[2]) / M[4,4])

        D = np.diag([M[0,0]/T1, M[1,1]/T2, M[2,2]/T3, M[3,3]*2*zeta4*w4, M[4,4]*2*zeta5*w5, M[5,5]/T6])
    else:
        G33 = hydrostatics[2,2]
        G44 = hydrostatics[3,3]
        G55 = hydrostatics[4,4]

        zeta3 = 0.2
        w3 = np.sqrt(G33 / M[2,2])
        w4 = np.sqrt(G44 / M[3,3])
        w5 = np.sqrt(G55 / M[4,4])

        D = np.diag([M[0,0]/T1, M[1,1]/T2, M[2,2]*2*zeta3*w3, M[3,3]*2*zeta4*w4, M[4,4]*2*zeta5*w5, M[5,5]/T6])

    return D

def Gmtrx(nabla, A_wp, GMT, GML, LCF, r_bp):

    rho = 1025
    g = 9.81

    r_bf = np.array([LCF, 0, 0])

    G33_CF = rho * g * A_wp
    G44_CF = rho * g * nabla * GMT
    G55_CF = rho * g * nabla * GML
    G_CF = np.diag([0, 0, G33_CF, G44_CF, G55_CF, 0])
    G_CO = Hmtrx(r_bf).T @ G_CF @ Hmtrx(r_bf)
    G = Hmtrx(r_bp).T @ G_CO @ Hmtrx(r_bp)

    return G

def m2c(M, nu):

    M = 0.5 * (M + M.T)

    if len(nu) == 6:
        M11 = M[0:3,0:3]
        M12 = M[0:3,3:6]
        M21 = M12.T
        M22 = M[3:6,3:6]

        nu1 = nu[0:3]
        nu2 = nu[3:6]
        nu1_dot = M11 @ nu1 + M12 @ nu2
        nu2_dot = M21 @ nu1 + M22 @ nu2

        C = np.block([[np.zeros((3,3)), -Smtrx(nu1_dot)],
                      [-Smtrx(nu1_dot), -Smtrx(nu2_dot)]])
    else:
        C = np.array([[0, 0, -M[1,1]*nu[1] - M[1,2]*nu[2]],
                      [0, 0, M[0,0]*nu[0]],
                      [M[1,1]*nu[1] + M[1,2]*nu[2], -M[0,0]*nu[0], 0]])

    return C