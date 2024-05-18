#!/usr/bin/env python3

# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

import mmgdynamics.calibrated_vessels as cvs
from dataclasses import dataclass
from mmgdynamics.maneuvers import *
from mmgdynamics.structs import Vessel, InitialValues

import time
import numpy as np

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
target_position = np.array([50,-50])
print("Target position set.")

# 환경 생성
def make_env(env_index):
    def _init():
        env = VesselEnv(vessel, initial_state, dT=0.1, target_position=target_position, render_mode='human', max_steps=200, slow_down_distance=10.0)
        env = Monitor(env, f"./logs/env_{env_index}")
        env.env_index = env_index
        return env
    return _init

# 환경 인스턴스 리스트 생성
num_envs = 4
envs = [make_env(i) for i in range(num_envs)]

# 벡터화된 환경 생성
vec_env = DummyVecEnv(envs)

print("Environment created and wrapped with Monitor.")
single_env = make_env(0)()
check_env(single_env)  # 이 함수 호출로 환경이 올바른지 검사

# 모델 훈련 및 평가
eval_callback = EvalCallback(vec_env, best_model_save_path="./logs/",
            log_path="./logs/", eval_freq=10000,
            deterministic=True, render=False, verbose=1)
print("Callback configured.")

# Logger 설정
log_path = "./logs/"
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

# PPO 모델 생성
model = PPO("MlpPolicy", vec_env, verbose=1)
model.set_logger(new_logger)
print("Model created.")

# 학습 시작 시간
start_time = time.time()
print("Training started at:", start_time)

# 모델 훈련
model.learn(total_timesteps=100000, callback=eval_callback)
print("Training completed.")

# 학습 시간 측정
end_time = time.time()
elapsed_time = end_time - start_time
print(f"학습 완료! 총 걸린 시간: {elapsed_time//60:.0f} 분 {elapsed_time%60:.0f} 초")

# 모델 저장
model.save("ppo_vessel_model")
print("Model saved.")

del model # remove to demonstrate saving and loading
