# src/usv_drl_project/utils/cpa_utils.py
import numpy as np
from config import CONFIG

def compute_cpa(own_state, target_state):
    x, y, psi, u = own_state["x"], own_state["y"], own_state["psi"], own_state["u"]
    pA = np.array([x, y])
    vA = np.array([
        u * np.cos(psi),
        u * np.sin(psi)
    ])

    pB = np.array([target_state['x'], target_state['y']])
    vB = np.array([
        target_state['u'] * np.cos(target_state['psi']),
        target_state['u'] * np.sin(target_state['psi'])
    ])

    dv = vA - vB
    dp = pA - pB
    norm_dv = np.linalg.norm(dv)

    if norm_dv <= 1e-5:
        tcpa = np.inf
    else:
        tcpa = -np.dot(dp, dv) / (norm_dv ** 2)

    cpa_A = pA + vA * tcpa
    cpa_B = pB + vB * tcpa
    dcpa = np.linalg.norm(cpa_A - cpa_B)

    return dcpa, tcpa

def is_risk(dcpa, tcpa, dcpa_thresh=30.0, tcpa_thresh=60.0, is_static=False):
    # print(f"[DEBUG is_risk] is_static={is_static}, dcpa={dcpa:.2f}, tcpa={tcpa:.2f}, "
    #   f"dcpa_thresh={dcpa_thresh}, tcpa_thresh={tcpa_thresh}")
    d_thresh = dcpa_thresh if dcpa_thresh is not None else CONFIG['dcpa_thresh']
    t_thresh = tcpa_thresh if tcpa_thresh is not None else CONFIG['tcpa_thresh']
    if is_static:
        return dcpa <= d_thresh
    return (0 <= tcpa <= t_thresh) and (dcpa <= d_thresh)
