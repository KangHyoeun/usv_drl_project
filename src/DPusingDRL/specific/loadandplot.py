#!/usr/bin/env python3

# loadandplot.py
from stable_baselines3 import PPO
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
target_position = np.array([100, 0])

# 환경 생성 함수
def make_env():
    def _init():
        env = VesselEnv(vessel, initial_state, dT=0.1, target_position=target_position, target_psi=0.0, render_mode='human', max_steps=1000, slow_down_distance=10.0)
        return env
    return _init

# 모델 로드
model = PPO.load("ppo_vessel_model")
print("Model loaded.")

# 단일 환경에서 모델 실행 및 시각화
env = make_env()()
obs = env.reset()
done = False
trajectory = []
rewards = []

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    trajectory.append(env.position.copy())
    rewards.append(reward)
    if done or truncated:
        break

# 시각화
trajectory = np.array(trajectory)
plt.figure(figsize=(10, 5))

# 궤적 플롯
plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
plt.scatter([target_position[0]], [target_position[1]], color='red', label='Goal')
plt.xlim(-10, 110)
plt.ylim(-20, 20)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Ship Trajectory')
plt.legend()

# 보상 플롯
plt.subplot(1, 2, 2)
plt.plot(rewards, label='Rewards')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Rewards Over Time')
plt.legend()

plt.tight_layout()
plt.show()

print("Simulation completed.")
