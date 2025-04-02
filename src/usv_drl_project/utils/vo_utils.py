# src/usv_drl_project/utils/vo_utils.py
import numpy as np
from config import CONFIG

def is_inside_vo(vA, pA, pB, rA, rB, vB):
    v_rel = vA - vB  # 상대 속도
    p_rel = pB - pA  # 상대 위치

    r_sum = rA + rB
    dist = np.linalg.norm(p_rel)
    if dist == 0:
        return True  # 완전 중첩

    p_dir = p_rel / dist
    projection = np.dot(v_rel, p_dir)  # 중심선 projection
    perp_dist = np.linalg.norm(v_rel - projection * p_dir)

    return perp_dist < (r_sum / dist)

def classify_velocity_region(v_rel, p_rel, cone_half_angle_deg=45):
    cone_half_angle_deg = CONFIG.get('cone_half_angle_deg', 45)  # config에서 반각도 값 가져오기

    if np.linalg.norm(v_rel) < 1e-5 or np.linalg.norm(p_rel) < 1e-5:
        return 'V3'

    # 기준 벡터: 상대 위치 방향
    p_dir = p_rel / np.linalg.norm(p_rel)
    v_rel_unit = v_rel / np.linalg.norm(v_rel)

    dot = np.dot(v_rel_unit, p_dir)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))  # [0, π]
    cross = np.cross(p_dir, v_rel_unit)  # 양수면 좌측, 음수면 우측

    # cone 반각도 기준
    cone_half = np.deg2rad(cone_half_angle_deg)
    if angle > cone_half:
        return 'V3'
    elif cross > 0:
        return 'V1'
    else:
        return 'V2'