# src/usv_drl_project/utils/guidance.py
import numpy as np
import math
from utils.angle import ssa
rA = rB = 2.0

def vector_field_guidance(y, chi_path=0.0, chi_inf=math.radians(60), k=1.0):
    return chi_inf * (2 / math.pi) * math.atan(k * y) + chi_path

def is_inside_vo(pA, vA, pB, vB):
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

def classify_velocity_region(pA, vA, pB, vB):
    vBA = vB - vA
    pAB = pB - pA

    in_vo = is_inside_vo(pA, vA, pB, vB)
    dot = np.dot(pAB, vBA)
    cross_z = np.cross(pAB, vBA)

    # V3: 후방 벗어남
    if dot < 0:
        return 'V3'
    
    # V1: VO 밖이고, V3도 아니고, 좌측
    if not in_vo and cross_z < 0:
        return 'V1'
    
    # V2: 그 외 (즉, 우측)
    return 'V2'

def select_avoidance_heading(own_state, target_state, direction='right', delta=np.radians(20)):
    x, y, psi, u = own_state["x"], own_state["y"], own_state["psi"], own_state["u"]

    pA = np.array([x, y])
    vA = np.array([u * math.cos(psi),
                    u * math.sin(psi)])
    pB = np.array([target_state['x'], target_state['y']])
    vB = np.array([target_state['u'] * math.cos(target_state['psi']),
                    target_state['u'] * math.sin(target_state['psi'])])

    # 금지속도 영역 내인지 확인
    if is_inside_vo(pA, vA, pB, vB):
        region = classify_velocity_region(pA, vA, pB, vB)
        # 충돌 위험한 방향일 경우 회피 방향 강제
        if direction == 'right' and region == 'V2':
            # 회피 속도 벡터 예시: 오른쪽 방향으로 틀기 (현재 속도에서 약간 오른쪽)
            avoid_angle = math.atan2(vA[1], vA[0]) - delta
        elif direction == 'left' and region == 'V1':
            avoid_angle = math.atan2(vA[1], vA[0]) + delta
        else:
            # 이미 안전한 방향이면 현재 속도 유지
            avoid_angle = math.atan2(vA[1], vA[0])
    else:
        # VO 밖이면 현재 방향 유지
        avoid_angle = math.atan2(vA[1], vA[0])

    return ssa(avoid_angle)  # χ_d

def get_available_avoid_directions(own_state, target_state):

    x, y, psi, u = own_state["x"], own_state["y"], own_state["psi"], own_state["u"]

    pA = np.array([x, y])
    vA = np.array([u * math.cos(psi),
                    u * math.sin(psi)])
    pB = np.array([target_state['x'], target_state['y']])
    vB = np.array([target_state['u'] * math.cos(target_state['psi']),
                    target_state['u'] * math.sin(target_state['psi'])])
    
    # 좌측 회피 시도
    v_left = rotate_vector(vA, math.radians(+20))
    left_safe = not is_inside_vo(pA, v_left, pB, vB)

    # 우측 회피 시도
    v_right = rotate_vector(vA, math.radians(-20))
    right_safe = not is_inside_vo(pA, v_right, pB, vB)

    return left_safe, right_safe

def rotate_vector(v, theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
