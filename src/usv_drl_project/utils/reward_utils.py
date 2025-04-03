import math

def compute_avoidance_reward(encounter_type, delta_angle_rad, tcpa=None):
    kr = -1.0
    # 조우 유형별 목표 회피각 또는 회피시점 기준 보상
    if encounter_type == 'HO':
        target_angle = math.radians(30)
        r = math.exp(kr * abs(delta_angle_rad - target_angle))

    elif encounter_type == 'SO':
        target_angle = math.radians(75)
        r = math.exp(kr * abs(delta_angle_rad - target_angle))

    elif encounter_type in ['GW', 'OT']:
        if tcpa is None:
            return 0.0
        tcpa_thresh = 10.0
        r = math.exp(kr * abs(tcpa - tcpa_thresh))

    elif encounter_type == 'Static':
        target_angle = math.radians(45)
        r = math.exp(kr * abs(delta_angle_rad - target_angle))

    else:
        return 0.0
    
    return r

def compute_path_reward(e_cross, krpath=-1.0):
    return math.exp(krpath * abs(e_cross))