#!/usr/bin/env python3
import numpy as np

def saturation(value, min_value, max_value):
    """
    x (float) = sat(value, min_value, max_value) saturates a signal x such that min_value <= min_value <= max_value
    """
    if value > max_value: 
        output = max_value
    elif value < min_value: 
        output =min_value
    else:
        output = value

    return output

class PIDControl:
    '''
    PIDControl():
        Class for PID contorl.
    '''
    def __init__(self):
        '''
        __init__: initalize parameters in class
            Inputs: 
                None
        '''
        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0
        self.prevError = None

    def update(self, kp, ki, kd):
        '''
        update: update parameter in class
        
        Inputs:
            kp (float): proportional gain for heading PID controller [-]
            
            ki (float): integral gain for heading PID controller [-]
            
            kd (float): derivative gain for heading PID controller [-]
        '''
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
    def output(self, error, saturation, dt):
        '''
        output: calcaulte output value of class
        
        Inputs:
            error (float): deviation between desired state and state [-]
            
            saturation (float): saturate the control value such as PWM, RPM and Force  [-]
            
            dt (float): sampling time [s]
                
        Outputs:
            controlValue (float): the output value of PID controller such as PWM, RPM and Force [-]
        '''
        if self.prevError is None:
            errorD = 0 
        else:
            errorD = (error - self.prevError) / dt
        errorI =+ error*dt 
        self.prevError = error
        controlValue = self.kp * error + self.ki * errorI + self.kd * errorD
        if saturation is None:
            pass
        else:
            if abs(controlValue) > saturation:
                controlValue = np.sign(controlValue) * saturation

        return controlValue
    
class Smoother:
    '''
    Smoother():
        Class for smoothing the value using second order differential equation, J.-Y. Park, 2016, Design of a Safety Operational Envelope Protection System for a Submerged Body.
    '''
    def __init__(self):
        '''
        __init__: initalize parameters in class
            Inputs: 
                None
        '''
        self.wn   = 0.0
        self.zeta = 0.0
        self.currentD      = 0.0
        self.current       = 0.0
        
    def update(self, wn, zeta):
        '''
        update: update parameter in class
        
        Inputs:
            wn (float): natural frequency
            
            zeta (float): damping ratio
        '''
        self.wn   = wn
        self.zeta = zeta

    def output(self, command, dt):
        '''
        output: calcaulte output value of class
        
        Inputs:
            command (float): deviation between desired state and state [-]
            
            dt (float): sampling time [s]
                
        Outputs:
            current (float): output value of smoother [-]
            currentD (float): first order differential value of output [-]
        '''
        currentData2D = self.wn**2*(command - self.current) - 2*self.zeta*self.wn*self.currentD
        self.currentD = self.currentD + currentData2D*dt
        self.current  = self.current + self.currentD * dt
        
        return self.current