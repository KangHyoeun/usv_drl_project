#!/usr/bin/env python3

# train.py
from stable_baselines3.common.monitor import Monitor

import mmgdynamics.calibrated_vessels as cvs
from dataclasses import dataclass

from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel, InitialValues
import time
import numpy as np
from stable_baselines3 import PPO
from callback import CustomEvalCallback
from customEnv import VesselEnv

@dataclass
class KVLCC2Inits:
    full_scale = InitialValues(
        u     = 3.85, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 1.05 # Propeller revs [s⁻¹]
    )
    
    l_64 = InitialValues(
        u     = 4.0, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 3.0 # Propeller revs [s⁻¹]
    )
    
    l_7 = InitialValues(
        u     = 1.128, # Longitudinal vessel speed [m/s]
        v     = 0.0, # Lateral vessel speed [m/s]
        r     = 0.0, # Yaw rate acceleration [rad/s]
        delta = 0.0, # Rudder angle [rad]
        nps   = 13.4 # Propeller revs [s⁻¹]
    )


# Use a pre-calibrated vessel
vessel = Vessel(**cvs.kvlcc2_l64)
ivs = KVLCC2Inits.l_64

# 초기 상태 정의
initial_state = np.array([ivs.u, ivs.v, ivs.r])
print("Initial state set.")

# 목표 위치 정의
target_position = np.array([10, 10])
print("Target position set.")

# 환경 생성
env = VesselEnv(vessel, initial_state, dT=0.1, target_position=target_position)
env = Monitor(env, "./logs/")
print("Environment created and wrapped with Monitor.")

# 모델 훈련 및 평가
callback = CustomEvalCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=1,
                             deterministic=True, render=True, verbose=1)
print("Callback configured.")

# PPO 모델 생성
model = PPO("MlpPolicy", env, verbose=1)
print("Model created.")

# 학습 시작 시간
start_time = time.time()
print("Training started at:", start_time)

# 모델 훈련
model.learn(total_timesteps=10, callback=callback)
print("Training completed.")

# 학습 시간 측정
end_time = time.time()
elapsed_time = end_time - start_time
print(f"학습 완료! 총 걸린 시간: {elapsed_time//60:.0f} 분 {elapsed_time%60:.0f} 초")

# 모델 저장
model.save("ppo_vessel_model")
print("Model saved.")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_vessel_model")
print("Model loaded.")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rew, done, trun, info = env.step(action)
    env.render("human")