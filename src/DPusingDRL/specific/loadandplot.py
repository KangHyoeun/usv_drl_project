#!/usr/bin/env python3

# loadandplot.py
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from customEnv import VesselEnv

# 초기 설정
import mmgdynamics.calibrated_vessels as cvs
from dataclasses import dataclass
from mmgdynamics.structs import Vessel, InitialValues

@dataclass
class KVLCC2Inits:
    full_scale = InitialValues(
        u     = 3.85,
        v     = 0.0,
        r     = 0.0,
        delta = 0.0,
        nps   = 1.05
    )
    
    l_64 = InitialValues(
        u     = 4.0,
        v     = 0.0,
        r     = 0.0,
        delta = 0.0,
        nps   = 3.0
    )
    
    l_7 = InitialValues(
        u     = 1.128,
        v     = 0.0,
        r     = 0.0,
        delta = 0.0,
        nps   = 13.4
    )

# 미리 보정된 선박 사용
vessel = Vessel(**cvs.kvlcc2_l64)
ivs = KVLCC2Inits.l_64

# 초기 상태 정의
initial_state = np.array([ivs.u, ivs.v, ivs.r])
target_position = np.array([50, -50])

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

# PPO 모델 생성
model = PPO("MlpPolicy", vec_env, verbose=1)
print("Model created.")

# 모델 로드
model = PPO.load("ppo_vessel_model")
print("Model loaded.")

obs = vec_env.reset()
done = [False] * num_envs

max_timesteps = 300  # 최대 타임스텝 수
current_timesteps = 0

while not all(done) and current_timesteps < max_timesteps:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    done = dones
    current_timesteps += 1
    for i, env in enumerate(vec_env.envs):
        env.unwrapped.render(render_mode='human', save_path='./plots', env_index=i)

if current_timesteps >= max_timesteps:
    print("Reached maximum timesteps.")
else:
    print("All environments completed.")