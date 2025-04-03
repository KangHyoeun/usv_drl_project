import numpy as np

def Smtrx(a):

    S = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]])

    return S

def Hmtrx(r):

    S = Smtrx(r)
    H = np.block([
        [np.eye(3), S.T],
        [np.zeros((3, 3)), np.eye(3)]
    ])
    return H

def Rzyx(phi, theta, psi):

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    R = np.array([[cpsi*cth, -spsi*cphi + cpsi*sth*sphi, spsi*sphi + cpsi*cphi*sth],
                  [spsi*cth, cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi],
                  [-sth, cth*sphi, cth*cphi]])

    return R

def Tzyx(phi, theta):

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)

    if np.isclose(cth, 0.0):
        raise ValueError("Tzyx is singular for theta = +-90 degrees")
    
    T = np.array([[1, sphi*sth/cth, cphi*sth/cth],
                  [0, cphi, -sphi],
                  [0, sphi/cth, cphi/cth]])

    return T

def eulerang(phi, theta, psi):

    J1 = Rzyx(phi, theta, psi)
    J2 = Tzyx(phi, theta)

    J = np.block([[J1, np.zeros((3,3))],
                  [np.zeros((3,3)), J2]])
    
    return J, J1, J2