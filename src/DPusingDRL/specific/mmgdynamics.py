#!/usr/bin/env python3
import matplotlib
import mmgdynamics.calibrated_vessels as cvs
from dataclasses import dataclass

from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel

@dataclass
class KVLCC2Inits:
    full_scale = InitialValues(
        u     = 3.85, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 1.05 # Propeller revs [s⁻¹]
    )
    
    l_64 = InitialValues(
        u     = 4.0, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 3.0 # Propeller revs [s⁻¹]
    )
    
    l_7 = InitialValues(
        u     = 1.128, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 13.4 # Propeller revs [s⁻¹]
    )


# Use a pre-calibrated vessel
vessel = Vessel(**cvs.kvlcc2_l7)

iters = 3000

class ManeuverNode:
    def __init__(self):
        self.dp = np.array([0., 0., 0.])
        self.dp_radius = 2.0 
        self.rate = 1

        self.ivs = KVLCC2Inits.l_7
        self.vessel = Vessel(**cvs.kvlcc2_l7)
        self.uvr = np.array([self.ivs.u, self.ivs.v, self.ivs.r])
        self.xypsi = np.array([0., 0., 0.])
        self.Udrift = np.array([0., 0.])

    def maneuver(self, uvr: np.ndarray, psi: float, delta: float, water_depth: float=None) -> np.ndarray:
        sol = step(X= uvr,
                   psi= psi,
                   vessel= self.vessel,
                   dT= 0.1,
                   nps= self.ivs.nps,
                   delta= delta_list[s],
                   fl_vel=None,
                   w_vel= 0,
                   beta_w= 0,
                   water_depth= water_depth)
        
        # Vel in x and y direction (m/s), angular turning rate (rad/s)
        u, v, r = sol

        # Transform to earth-fixed coordinate system

        psi += r
        print(s, round(float(r), 4), psi*180/math.pi)
        v_x = math.cos(psi) * u - math.sin(psi) * v
        v_y = math.sin(psi) * u + math.cos(psi) * v

        if s == 0:
            res[0, s] = v_x
            res[1, s] = v_y
        else:
            res[0, s] = res[0, s-1] + v_x
            res[1, s] = res[1, s-1] + v_y

        U = math.sqrt(u**2+v**2)
        drift = -math.atan(-v/u)*180/math.pi
        res[2, s] = drift

        # Set current solution as next initial values
        uvr = np.hstack(sol)

        return res
        


