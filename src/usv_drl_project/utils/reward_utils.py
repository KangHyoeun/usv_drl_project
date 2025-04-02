import math

def compute_avoidance_reward(encounter_type, delta_angle_rad, tcpa=None, env_bias=None):
    kr = 1.0
    if encounter_type == 'HO':
        target_angle = math.radians(30)
    elif encounter_type == 'SO':
        target_angle = math.radians(75)
    elif encounter_type == 'GW':
        if tcpa is None:
            return 0.0
        tcpathresh = 10.0
        return math.exp(-kr * abs(tcpa - tcpathresh))
    elif encounter_type in ['OT', 'Static']:
        target_angle = math.radians(45)
    else:
        return 0.0

    r = math.exp(-kr * abs(delta_angle_rad - target_angle))
    if env_bias is not None:
        r *= math.exp(-env_bias)
    return r
