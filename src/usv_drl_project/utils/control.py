# utils/control.py
import math
import numpy as np
from utils.angle import ssa

def steering_controller(chi_d, psi, prev_psi, dt, kp=1.0, kd=0.2):
    """
    PD 제어기로 delta(rpm 차이)를 계산.
    - chi_d: 목표 침로각
    - psi: 현재 선수각
    - prev_psi: 이전 timestep의 선수각
    - dt: 시간 간격
    """
    psi_error = ssa(chi_d - psi)
    d_psi = ssa(psi - prev_psi) / dt if prev_psi is not None else 0.0
    delta = kp * psi_error - kd * d_psi
    return np.clip(delta, -0.5, 0.5)
