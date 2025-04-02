# src/usv_drl_project/utils/cpa_utils.py
import numpy as np
from config import CONFIG

def compute_cpa(own_state, target_state):
    pA = np.array([own_state['x'], own_state['y']])
    vA = np.array([
        own_state['u'] * np.cos(own_state['psi']),
        own_state['u'] * np.sin(own_state['psi'])
    ])

    pB = np.array([target_state['x'], target_state['y']])
    vB = np.array([target_state.get('vx', 0.0), target_state.get('vy', 0.0)])

    dv = vA - vB
    dp = pA - pB
    norm_dv = np.linalg.norm(dv)

    if norm_dv <= 1e-5:
        tcpa = 0.0
    else:
        tcpa = -np.dot(dp, dv) / (norm_dv ** 2)

    cpa_A = pA + vA * tcpa
    cpa_B = pB + vB * tcpa
    dcpa = np.linalg.norm(cpa_A - cpa_B)

    if is_risk(dcpa, tcpa):  # config의 임계값을 이용하여 위험 여부 확인
        return dcpa, tcpa, True  # 위험이 있으면 True 반환
    return dcpa, tcpa, False  # 위험이 없으면 False 반환

def is_risk(dcpa, tcpa, dcpa_thresh=10.0, tcpa_thresh=20.0):
    dcpa_thresh = dcpa_thresh if dcpa_thresh is not None else CONFIG['dcpa_thresh']
    tcpa_thresh = tcpa_thresh if tcpa_thresh is not None else CONFIG['tcpa_thresh']
    return (0 <= tcpa <= tcpa_thresh) and (dcpa <= dcpa_thresh)
