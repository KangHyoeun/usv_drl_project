# src/usv_drl_project/utils/guidance.py
import math

def vector_field_guidance(y, chi_path=0.0, chi_inf=math.radians(45), k=1.0):
    return chi_inf * (2 / math.pi) * math.atan(k * y) + chi_path
