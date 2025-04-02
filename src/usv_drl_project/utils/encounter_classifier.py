# src/usv_drl_project/utils/encounter_classifier.py
import numpy as np
import math

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_relative_angle(own_heading, target_pos, own_pos):
    dx = target_pos[0] - own_pos[0]
    dy = target_pos[1] - own_pos[1]
    rel_angle = math.atan2(dy, dx)
    return wrap_to_pi(rel_angle - own_heading)

def get_relative_heading(own_heading, target_heading):
    return wrap_to_pi(target_heading - own_heading)

def classify_encounter(own_state, target_state):
    """
    Returns: HO / OT / GW / SO / SF (Tam & Bucknall 기반)
    """
    os_pos = (own_state['x'], own_state['y'])
    os_heading = own_state['psi']

    ts_pos = (target_state['x'], target_state['y'])
    ts_heading = target_state.get('psi', 0.0)  # 정적 장애물은 default 0

    rel_angle = get_relative_angle(os_heading, ts_pos, os_pos)  # 장애물의 상대 위치
    rel_heading = get_relative_heading(os_heading, ts_heading)  # 장애물의 상대 선수각

    # 6구역 R1~R6 (상대 위치 기준)
    sector = None
    if -np.pi/8 <= rel_angle < np.pi/8:
        sector = 'R1'
    elif np.pi/8 <= rel_angle < 3*np.pi/8:
        sector = 'R2'
    elif 3*np.pi/8 <= rel_angle < 5*np.pi/8:
        sector = 'R3'
    elif 5*np.pi/8 <= rel_angle < 7*np.pi/8:
        sector = 'R4'
    elif -7*np.pi/8 <= rel_angle < -5*np.pi/8:
        sector = 'R5'
    elif -5*np.pi/8 <= rel_angle < -3*np.pi/8:
        sector = 'R6'
    else:
        sector = 'R1'  # 기본값

    # 상대 선수각 TSR 분류 → TSR1~TSR6 생략하고 통합
    encounter_type = 'SF'
    if sector == 'R1':
        encounter_type = 'HO'
    elif sector == 'R2':
        encounter_type = 'HO'
    elif sector == 'R3':
        encounter_type = 'GW'
    elif sector == 'R4':
        encounter_type = 'OT'
    elif sector == 'R5':
        encounter_type = 'OT'
    elif sector == 'R6':
        encounter_type = 'SO'

    if not target_state.get('dynamic', True):
        encounter_type = 'OT'  # 정적 장애물은 임의 회피 허용

    return encounter_type
