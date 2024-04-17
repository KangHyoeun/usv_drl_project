#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
def ssa(angle):
    """
    angle (float) = ssa(angle) returns the smallest-signed angle in [ -pi, pi ]
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
        
    return angle

@dataclass
class LowPassFilter:
    '''
    Class for implementing Low Pass Filter(LPF) algorithm, which filters out high frequencies (noise) and lets low frequencies through.
    '''
    pose: np.ndarray  = None    # [x, y, psi]
    vel: np.ndarray  = np.zeros(3)  # [u, v, r]

    def update_velocity(self, pose:np.ndarray, rot , dt:float , weight:float=0.0)->np.ndarray : 
        '''
        Inputs: 
            pose [x (m), y (m), psi (deg)] : position(x,y) and yaw (earth fix coordinate) 
                
            dt (sec) : system cycle시스템 주기

            weight (-) : Impact of previous values (0 <= weight <= 1)

        Outputs:
            vel  [u (m/s), v (m/s), r (deg/s)] : low frequencies velocity data (body fix coordinate)
        '''
        if self.pose is None:
            self.pose = pose
            return self.vel

        psi = self.pose[2]
        R = np.array([[np.cos(psi), np.sin(psi), 0],
                    [-np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])
        u = (pose[0] - self.pose[0]) / dt
        v = (pose[1] - self.pose[1]) / dt
        r = ssa(pose[2] - self.pose[2]) / dt
        r = rot
        new_vel = np.array([u, v, r])
        self.pose = pose
        self.vel = (1 - weight) * new_vel + weight * self.vel
        
        body_fix_vel = np.dot(R, self.vel.T)  
        return body_fix_vel

class KalmanFilter:
    def __init__(self):
        self.dt = 0.1
        self.x = np.matrix(np.zeros((9, 1)))
        self.A = np.matrix([1, 0, 0, self.dt, 0, 0, 0, 0, 0, 
                            0, 1, 0, 0, self.dt, 0, 0, 0 ,0,
                            0, 0, 1, 0, 0, self.dt, 0, 0, 0,
                            0, 0, 0, 1, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 1, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 1, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(9,9)

        self.B = np.matrix([0.5*self.dt, 0, 0, 0, 0, 0,
                            0, 0.5*self.dt, 0, 0, 0, 0,
                            0, 0, 0.5*self.dt, 0, 0, 0,
                            self.dt, 0, 0, 0, 0, 0,
                            0, self.dt, 0, 0, 0, 0,
                            0, 0, self.dt, 0, 0, 0,
                            0, 0, 0, self.dt, 0, 0,
                            0, 0, 0, 0, self.dt, 0,
                            0, 0, 0, 0, 0, self.dt]).reshape(9,6)

        self.I_jac = np.matrix([0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0,
                                0, 1, 0, 0, 0, 0,
                                0, 0, 1, 0, 0, 0,
                                0, 0, 0, 1, 0, 0,
                                0, 0, 0, 0, 1, 0,
                                0, 0, 0, 0, 0, 1]).reshape(9,6)
        
        self.H_jac = np.matrix([1, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 1, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 1, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 1, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 1, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 1, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 1, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 1, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(9,9)
        
        self.P = np.matrix(np.eye(9, 9))
        self.I = np.matrix(np.eye(9, 9))
    
    def update(self, u, Q, z, R):
        
        # state predict
        x = np.dot(self.A, self.x) + np.dot(self.B, u)
        P = np.dot(np.dot(self.A, self.P), self.A.T) + np.dot(np.dot(self.I_jac, Q), self.I_jac.T)

        # update
        K = np.dot(np.dot(self.P, self.H_jac.T), np.linalg.inv(np.dot(np.dot(self.H_jac, self.P), self.H_jac.T) + R))

        self.x = x + np.dot(K, z - np.dot(self.H_jac, x))
        self.P = np.dot(self.I - np.dot(K, self.H_jac), P)
