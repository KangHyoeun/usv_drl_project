# src/usv_drl_project/utils/vo_utils.py
import numpy as np
import math

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

def classify_velocity_region(vA, pA, vB, pB, rA, rB):
    vBA = vB - vA
    pAB = pB - pA

    in_vo = is_inside_vo(vA, pA, pB, rA, rB, vB)
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

def select_avoidance_heading(vA, pA, pB, vB, direction='right'):
    rA = rB = 2.0

    # 금지속도 영역 내인지 확인
    if is_inside_vo(vA, pA, pB, rA, rB, vB):
        region = classify_velocity_region(vA, pA, pB, rA, rB, vB)
        # 충돌 위험한 방향일 경우 회피 방향 강제
        if direction == 'right' and region == 'V2':
            # 회피 속도 벡터 예시: 오른쪽 방향으로 틀기 (현재 속도에서 약간 오른쪽)
            avoid_angle = math.atan2(vA[1], vA[0]) - math.radians(30)
        elif direction == 'left' and region == 'V1':
            avoid_angle = math.atan2(vA[1], vA[0]) + math.radians(30)
        else:
            # 이미 안전한 방향이면 현재 속도 유지
            avoid_angle = math.atan2(vA[1], vA[0])
    else:
        # VO 밖이면 현재 방향 유지
        avoid_angle = math.atan2(vA[1], vA[0])

    return avoid_angle  # χ_d

def get_available_avoid_directions(vA, pA, pB, vB):
    rA = rB = 2.0
    # 좌측 회피 시도
    v_left = rotate_vector(vA, math.radians(+30))
    left_safe = not is_inside_vo(v_left, pA, pB, rA, rB, vB)

    # 우측 회피 시도
    v_right = rotate_vector(vA, math.radians(-30))
    right_safe = not is_inside_vo(v_right, pA, pB, rA, rB, vB)

    return left_safe, right_safe

def rotate_vector(v, theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
