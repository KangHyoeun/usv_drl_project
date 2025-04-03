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
    elif np.pi/8 <= rel_angle < np.pi/2:
        sector = 'R2'
    elif np.pi/2 <= rel_angle < 5*np.pi/8:
        sector = 'R3'
    elif 5*np.pi/8 <= rel_angle < np.pi or -np.pi <= rel_angle < -5*np.pi/8:
        sector = 'R4'
    elif -5*np.pi/8 <= rel_angle < np.pi/2:
        sector = 'R5'
    elif np.pi/2 <= rel_angle < -np.pi/8:
        sector = 'R6'
    else:
        sector = 'R1'  # 기본값

    # TSR1~TSR6 구분 
    tsr = None
    if -3*np.pi/8 <= rel_heading <= 3*np.pi/8:
        tsr = 'TSR1'
    elif 3*np.pi/8 < rel_heading <= np.pi/2:
        tsr = 'TSR2'
    elif np.pi/2 < rel_heading <= 7*np.pi/8:
        tsr = 'TSR3'
    elif 7*np.pi/8 < rel_heading <= np.pi or -np.pi <= rel_heading < -7*np.pi/8:
        tsr = 'TSR4'
    elif -7*np.pi/8 <= rel_heading < -np.pi/2:
        tsr = 'TSR5'
    elif -np.pi/2 <= rel_heading < -3*np.pi/8:
        tsr = 'TSR6'
    else:
        tsr = 'TSR1'  # fallback

    # R + TSR 조합 기반 조우상황 결정 
    encounter_map = {
        ('R1', 'TSR4'): 'HO',
        ('R1', 'TSR2'): 'SO',
        ('R1', 'TSR3'): 'SO',
        ('R1', 'TSR1'): 'OT',
        ('R1', 'TSR5'): 'GW',
        ('R1', 'TSR6'): 'GW',

        ('R2', 'TSR4'): 'HO',
        ('R2', 'TSR1'): 'OT',
        ('R2', 'TSR5'): 'GW',
        ('R2', 'TSR6'): 'GW',

        ('R3', 'TSR1'): 'OT',
        ('R3', 'TSR5'): 'GW',
        ('R3', 'TSR6'): 'GW',

        ('R6', 'TSR4'): 'HO',
        ('R6', 'TSR2'): 'SO',
        ('R6', 'TSR3'): 'SO',
        ('R6', 'TSR1'): 'OT',

        ('R4', 'TSR2'): 'SO',
        ('R4', 'TSR1'): 'OT',
        ('R4', 'TSR6'): 'GW',

        ('R5', 'TSR2'): 'SO',
        ('R5', 'TSR3'): 'SO',
        ('R5', 'TSR1'): 'OT',
    }

    encounter_type = encounter_map.get((sector, tsr), 'SF')

    if not target_state.get('dynamic', True):
        encounter_type = 'Static' 

    return encounter_type
